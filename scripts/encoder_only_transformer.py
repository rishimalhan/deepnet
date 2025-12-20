#! /usr/bin/env python3

from argparse import ArgumentParser
import numpy as np
import math
import os
from sympy.series import sequences
import torch
from typing import Dict
import json
from torch.nn import Module, Linear, LayerNorm, Embedding
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from IPython import embed
from transformers import AutoTokenizer

DIR_PATH = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(DIR_PATH, "data")
MODEL_DIR = os.path.join(DIR_PATH, "model")
EMB_SIZE = 256
FILE_NAME = "space_set.txt"
TOKENIZER_MODEL = "gpt2"
EPOCH = 200
POSITIONAL_EMBEDDING_LEN = 512
BATCH_SIZE = 8
DEVICE = "mps"
LEARNING_RATE = 1e-5


class TokenizerFactory:
    _instances = {}

    @classmethod
    def get(cls, model_name: str):
        if model_name not in cls._instances:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            cls._instances[model_name] = tokenizer
        return cls._instances[model_name]


class SpaceJsonlInstructionDataset(Dataset):
    def __init__(self, path: os.PathLike):
        self._tokenizer = TokenizerFactory.get(TOKENIZER_MODEL)
        self._samples = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                self._samples.append(
                    {
                        "input": payload.get("input").strip(),
                        "output": payload.get("output").strip(),
                    }
                )

    def __len__(self) -> list:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self._samples[idx]

    def collate_fn(self, samples: list) -> list:
        T_max = -1
        seqs = []
        prompt_lengths = []
        for sample in samples:
            input_ids = self._tokenizer(
                text=sample.get("input"), add_special_tokens=False
            ).get("input_ids")
            output_ids = self._tokenizer(
                text=sample.get("output"), add_special_tokens=False
            ).get("input_ids")
            concatenated_ids = torch.LongTensor(
                input_ids
                + [self._tokenizer.eos_token_id]
                + output_ids
                + [self._tokenizer.eos_token_id]
            )
            if len(concatenated_ids) > T_max:
                T_max = len(concatenated_ids)
            seqs.append(concatenated_ids)
            prompt_lengths.append(len(input_ids) + 1)
        padded_seq = pad_sequence(
            sequences=seqs,
            batch_first=True,
            padding_value=self._tokenizer.pad_token_id,
        )
        attention_mask = torch.zeros_like(padded_seq)
        attention_mask[padded_seq != self._tokenizer.pad_token_id] = 1
        labels = padded_seq.clone()
        labels[:, :-1] = padded_seq[:, 1:]
        labels[:, -1] = -100
        for i, p in enumerate(prompt_lengths):
            labels[i, : max(p - 1, 0)] = -100
        return padded_seq, attention_mask, labels


class Transformer(Module):
    def __init__(self, embedding_dim, d_model, ff_dim):

        super(Transformer, self).__init__()
        self._d_model = d_model
        self._tokenizer = TokenizerFactory.get(TOKENIZER_MODEL)
        # Embedding to Q, K, V. (B, T, embedding_dim) to (B, T, d_model)
        self._query = Linear(in_features=embedding_dim, out_features=d_model)
        self._key = Linear(in_features=embedding_dim, out_features=d_model)
        self._value = Linear(in_features=embedding_dim, out_features=d_model)
        self._layernorm1 = LayerNorm(d_model)
        # Multi layer perceptron
        self._layer1 = Linear(in_features=d_model, out_features=ff_dim)
        self._layer2 = Linear(in_features=ff_dim, out_features=ff_dim)
        self._layer3 = Linear(in_features=ff_dim, out_features=len(self._tokenizer))
        self._embeddings = Embedding(
            embedding_dim=EMB_SIZE,
            num_embeddings=len(self._tokenizer),
            device=DEVICE,
            padding_idx=self._tokenizer.pad_token_id,
        )
        self._position_embeddings = Embedding(
            embedding_dim=EMB_SIZE,
            num_embeddings=POSITIONAL_EMBEDDING_LEN,
            device=DEVICE,
        )

    def forward(self, x, attention_masks):
        B, T = x.shape
        if T > POSITIONAL_EMBEDDING_LEN:
            raise ValueError(
                "The length of tokens is more than positional embedding length"
            )
        pos_ids = torch.arange(T, device=x.device)
        pos_emb = self._position_embeddings(pos_ids).unsqueeze(0)  # (1, T, emb_size)
        x = self._embeddings(x) + pos_emb  # (B, T, EMB_SIZE)
        Q = self._query(x)  # (B, T, d_model)
        K = self._key(x)  # (B, T, d_model)
        V = self._value(x)  # (B, T, d_model)
        attention_scores = Q @ K.transpose(-2, -1)  # (B, T, T)
        reshaped_masks = attention_masks.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(
            reshaped_masks == 0, float("-inf")
        )  # -inf because applying softmax this becomes 0 probability
        T = attention_scores.size(-1)
        causal = torch.triu(
            torch.ones(T, T, device=attention_scores.device, dtype=torch.bool),
            diagonal=1,
        )
        attention_scores = attention_scores.masked_fill(causal, float("-inf"))
        normalized_attention_scores = attention_scores / math.sqrt(self._d_model)
        attention_weights = F.softmax(normalized_attention_scores, dim=-1)

        # Scale value according to weights
        scaled_value = attention_weights @ V  # (B, T, d_model)
        out = self._layernorm1(scaled_value)  # (B, T, d_model)
        out = F.relu(self._layer1(out))  # (B, T, ff_dim)
        out = F.relu(self._layer2(out))  # (B, T, ff_dim)
        logits = self._layer3(out)  # (B, T, vocab_size)
        return logits


def load_model(ckpt_path: str = "") -> Transformer:
    model = Transformer(
        embedding_dim=EMB_SIZE, d_model=EMB_SIZE, ff_dim=4 * EMB_SIZE
    ).to(DEVICE)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
    return model


@torch.no_grad()
def generate(model: Transformer, prompt: str, max_new_tokens: int = 64) -> str:
    tokenizer = model._tokenizer
    eos_id = tokenizer.eos_token_id
    # Tokenize prompt
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = torch.LongTensor([prompt_ids + [eos_id]]).to(DEVICE)  # [1, T]

    for _ in range(max_new_tokens):
        attention_masks = torch.ones_like(input_ids, device=DEVICE)  # [1, T]
        logits = model(input_ids, attention_masks)  # [1, T, V]
        next_logits = logits[:, -1, :]  # [1, V] -> predict next token after last
        probs = torch.softmax(next_logits / 0.9, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id[:, :]], dim=1)  # append -> [1, T+1]
        if next_id.item() == eos_id:
            break
    # Decode only the generated continuation (after the prompt+eos)
    gen_ids = input_ids[0].tolist()
    # We inserted eos after prompt, so continuation starts after that first eos
    # Find first eos index:
    try:
        split = gen_ids.index(eos_id) + 1
    except ValueError:
        split = len(prompt_ids)
    continuation = gen_ids[split:]
    return (
        "Input:\t"
        + prompt
        + "\nResponse:\t"
        + tokenizer.decode(continuation, skip_special_tokens=True)
    )


def train():
    dataset = SpaceJsonlInstructionDataset(path=os.path.join(DATA_DIR, FILE_NAME))
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=False,
        collate_fn=dataset.collate_fn,
    )
    model = load_model()
    optim = Adam(params=model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCH):
        losses = []
        model.train()
        for i, (sequences, attention_masks, labels) in enumerate(dataloader):
            sequences = sequences.to(DEVICE)
            attention_masks = attention_masks.to(DEVICE)
            labels = labels.to(DEVICE)
            optim.zero_grad()
            logits = model.forward(x=sequences, attention_masks=attention_masks)
            output_distribution = logits.view(-1, len(model._tokenizer))
            labels = labels.masked_fill(attention_masks == 0, -100)
            loss = F.cross_entropy(
                output_distribution, labels.view(-1), ignore_index=-100
            )
            losses.append(loss.item())
            loss.backward()
            optim.step()
        print(f"Epoch # {epoch}. Average loss: {np.mean(losses):.3f}")
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "tiny_transformer.pth"))


if __name__ == "__main__":
    parser = ArgumentParser(prog="Transformer")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if args.train:
        train()
    else:
        model = load_model(os.path.join(MODEL_DIR, "tiny_transformer.pth"))
        exception = None
        while not isinstance(exception, KeyboardInterrupt):
            try:
                prompt = input("Enter a prompt: ")
                print(generate(model, prompt))
            except Exception as e:
                exception = e
