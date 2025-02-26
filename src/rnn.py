#! /usr/bin/python3

"""
A statistical language model is probability distribution over a sequence of words
P(wt | V) where t C V
Recurrent Neural Networks:
- Sequence models deals with ordered data where placement of data matters
- Deal with temporal aspect of data
- Can write similar to the data trained on
- Applications: Sentence completion, time series prediction, machine translation, summarizing

Word embedding
- Convert language to float that neural network can understand
- All unique words in the problem are vocabulary
- On hot embedding will take all words in the vocab and create vector of 1s and 0s.
    - 1 for index of the word
- Length of vector is length of vocabulary
- Naturally this is memory intensive and also doesn't capture relationship or contextual understanding
    between words
- Word embedding on the other hand would have a fixed embedding length
- Each dimension captures some characteristic e.g. male <> king
- Efficient and dense matrix
- Word2Vec, GloVe, FastText are popular word embeddings

Intuition
RNNs can capture intuition in the language
e.g. Apple is a fruit. It is red in color.
Here intuition is ability to answer Which fruit are we talking about? Which fruit is red in color?

RNNs
- Have a way to store previous information
- Internal state that is passed on to the next time step i.e. hidden state
- Given xt input, ut as input weight, wt as output weight, vt as feedback, ht-1 as previous hidden state
- ht = sigma(vt*ht-1 + ut*xt)
- ot = wt*ht

Sizes:
- Batch size: number of bptts passed
- bptt: batch propagation through time i.e. number of sentences in a sequence in a batch
- x is bxe where b is batch size and e is embedding size
- output is bxv where v is vocabulary size. So we can expect a sequence of words as output

Types:
One-One: Vanilla RNN
One-Many: Image captioning
Many-One: Sentiment analysis
Many-Many: Machine translation
Many-Many: Video classification

Layers of RNN can also be stacked

Two typical problems:
- Vanishing gradients: Solved by advanced architectures like LSTM and GRU
- Exploding gradients: Solved by clipping the gradients
"""

import os
import torch
import IPython
from torch.autograd import Variable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEVICE = "mps" if torch.mps.is_available() else "cpu"
SHAKESPEARE_DIR = os.path.join(ROOT, "shakespeare/")

class Dictionary(object):
        def __init__(self):
            self.word2idx = {}
            self.idx2word = []

        def add_word(self, word):
            if word not in self.word2idx:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1
            return self.word2idx[word]

        def __len__(self):
            return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()

        # This is very english language specific
        # We will ingest only these characters:
        self.whitelist = [chr(i) for i in range(32, 127)]

        self.train = self.tokenize(os.path.join(path, "train.txt"))
        self.valid = self.tokenize(os.path.join(path, "valid.txt"))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, "r", encoding="utf8") as f:
            tokens = 0
            for line in f:
                line = "".join([c for c in line if c in self.whitelist])
                words = line.split() + ["<eos>"]
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                line = "".join([c for c in line if c in self.whitelist])
                words = line.split() + ["<eos>"]
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

class RNNModel(torch.nn.Module):
    def __init__(
        self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5
    ):
        super(RNNModel, self).__init__()

        self.encoder = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_size
        )
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.rnn = torch.nn.GRU(
            embed_size, hidden_size, num_layers, dropout=dropout
        )
        self.dropout2 = torch.nn.Dropout(p=dropout)
        self.decoder = torch.nn.Linear(hidden_size, vocab_size)

        self._init_encoder_weights()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, input, hidden):
        emb = self.dropout1(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.dropout2(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        return (
            decoded.view(output.size(0) * output.size(1), decoded.size(1)),
            hidden,
        )

    def _init_encoder_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return Variable(
            weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        )

def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data

def get_batch(source, i, bptt_size, evaluation=False):
    seq_len = min(bptt_size, len(source) - 1 - i)
    data = Variable(source[i : i + seq_len], volatile=evaluation)
    target = Variable(source[i + 1 : i + 1 + seq_len].view(-1))
    return data.to(DEVICE), target.to(DEVICE)

path = os.path.join(ROOT, "data", "shakespear")
corpus = Corpus(path)

bs_train = 20  # batch size for training set
bs_valid = 10  # batch size for validation set
bptt_size = (
    35  # number of times to unroll the graph for back propagation through time
)
clip = 0.25  # gradient clipping to check exploding gradient

embed_size = 200  # size of the embedding vector
hidden_size = 200  # size of the hidden state in the RNN
num_layers = 2  # number of RNN layres to use
dropout_pct = 0.5  # %age of neurons to drop out for regularization

train_data = batchify(corpus.train, bs_train)
val_data = batchify(corpus.valid, bs_valid)

vocab_size = len(corpus.dictionary)

model = RNNModel(vocab_size, embed_size, hidden_size, num_layers, dropout_pct).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()

data, target = get_batch(train_data, 1, bptt_size)

def train(data_source, lr, bptt_size, bs_train, clip=0.25):
    # Turn on training mode which enables dropout.

    model.train()
    total_loss = 0
    hidden = model.init_hidden(bs_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for batch, i in enumerate(range(0, data_source.size(0) - 1, bptt_size)):
        data, targets = get_batch(data_source, i, bptt_size)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = Variable(hidden.data).to(DEVICE)

        # model.zero_grad()
        optimizer.zero_grad()

        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, vocab_size), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)

        optimizer.step()
        total_loss += len(data) * loss.data

    return total_loss / len(data_source)

def evaluate(data_source, bptt_size, bs_valid, vocab_size):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(bs_valid)

    for i in range(0, data_source.size(0) - 1, bptt_size):
        data, targets = get_batch(data_source, i, bptt_size, evaluation=True)

        output, hidden = model(data, hidden)
        output_flat = output.view(-1, vocab_size)

        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = Variable(hidden.data)

    return total_loss / len(data_source)

best_val_loss = None

def run(epochs, lr, bptt_size, bs_train, bs_valid, vocab_size):
    global best_val_loss

    for epoch in range(0, epochs):
        train_loss = train(train_data, lr, bptt_size, bs_train)
        val_loss = evaluate(val_data, bptt_size, bs_valid, vocab_size)
        logger.info(f"Epoch: {epoch}. Train Loss: {train_loss}. Valid Loss: {val_loss}")

        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(ROOT, "model", "rnn.model.pth"))

run(100, 0.001, bptt_size, bs_train, bs_valid, vocab_size)

IPython.embed()