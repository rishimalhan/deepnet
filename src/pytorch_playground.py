#! /usr/bin/python3

"""
Based off Udemy course
Deep Learning with Pytorch - Anand Saha

SECTION-1
* Introduction
Pytorch tensors are like numpy but on GPUs
Pytorch is dynamic whereas tensorflow is static
Pytorch recreates the graph from scratch in every iteration
Dynamic approach uses imperative style of programming whereas static uses declarative
Linear thought process. Faster and easier to debug.
Stacktrace gives exact line of error.

* Installing
Windows anaconda
CUDA: Parallel computing platform and programming model
Deeplearning or other app frameworks donot directly talk to GPU but use CUDA

* Tensors
Scalar (1,) Vector (N,) Matrix (N,M) Tensor (P,Q,R)
Rank in this context is length of dimension array or indices used to access an element
Tensor is multi-dim matrix of single data type, like np.ndarray
Stored on CPU or GPU

-Definitions
torch.Tensor class
Interoperable with numpy arrays
Tensor methods like size(), dim() similar to numpy
Create tensor from another tensor or python list
Cast tensor t into another using t.type(tensor.IntTensor)
Create tensor from nump array
Methods like t1.ones_like(t2) creates ONE tensor of same dimension as t2

-Operations
Inplace/Out-of-Place
Add: t = t1.add(t2) t1 and t2 do not get modified | t1.add_(t2) t1 gets modified
Cosine: t1.cos() or torch.cos(t1)
torch.linspace(0, 5, step=1) Others: arange, rand (uniform), randn (normal)
torch.mm : matrix multiplication

* Differentiation
y = f(x) = 2x | f(x)-Model and 2 is model parameter
In learning we are given x and y as data and goal is to find model parameter

* Computation graph
Graph of variables and operations conducted on those variables
Node is a variable or operation
NN are expressed as computation graphs
Using backpropagation and diff calculus we train model on desired I/O characteristics
Pytorch uses dynamic computation graphs as opposed to tensorflow static computation graph

* Variables
torch.autograd
Variables hold tensor for data. Parent who created the tensor. Gradient of Output wrt to tensor
Gradient of y wrt x is stored in variable x once y.backward() is called

* Interoperatability between numpy and pytorch
Memory pointers are shared between Pytorch and Numpy
torch.from_numpy(np.ndarray). Here changes in numpy will result in changes in tensor

* Torch on GPU
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name(0)
t = torch.FloatTensor()
t.cuda(0) means 0 number device
t.cpu() takes it back to CPU

* Terminology
Dataset is entire corpus of training inputs
Epoch is One pass through dataset
Batch is one portion of dataset
Iteration is a pass through batch
A pass would mean forward and backward propagation
e.g. 1000 image Epoch could be 50 Batch and 20 Iterations

CIFAR10 is a popular dataset. 60000 datapoints with 10 classes
torchvision allows to access the dataset
torch.utils.data.Dataset is abstract class used to access datasets and __iter__ and
    __len__ allows functions to iterate through datasets
transform.compose(*set of transforms*) is one such example
torch.utils.data.DataLoader provides interface to manipulate data...
    fetch batches at a time, shuffle, and load parallel data

* Tasks using Deep learning
Having computer to make sense of pixels and classify and localize
Sentiment analysis: Subjective text input and determine sentiment
Translation
Deep learning is ML technique to learn from abstract, raw, high dimension data
ML we need to feature engineer, and select transform/representations on data
    e.g. Human can't know what makes a Cat/Dog
DL learns even the features and then determines hierarchy of transforms


SECTION-2
* Simple Neural Network
- ANN can be viewed as a graph with nodes and edges
- Activation function can be viewed as a kink or transformation
- Activation function is also known as transfer function
- Sigmoid has a nice gradient in around X=0 and then approaches hard 0 or 1 value
- tanh and ReLU are others

* Loss Function
To determine loss score or how good prediction from the model is
Pytorch provides a set of loss functions: crossentropy, MSE, L1, BCE, etc

* Optimizer
Find best solution from possible solutions given some criteria
In our case, we want best values for model parameters
SGD, Adam, Adagrad, Adadelta, RMSprop
Learning rate tells optimizer how big step size should be
nesterov momentum helps converge SGD faster by taking into account past gradients
The weight of previous gradients exponentially reduces as optimization progresses


* Training network
Loop through epoch
    Loop through iterations
Make sure dataloader values provided by custom dataset are in torch Variables type
Put network in training mode (There is also eval mode). Dropout is not present in eval for e.g.
Clear off older gradients

* Save and Load
Save and load is feasible using state_dict() method of model

* Train on GPU
Move model during initialization and all train/test tensors to cuda within the loops


SECTION-3
* Computer Vision
Computers obtaining high level understanding from images and videos
Making computers do tasks that human visual system can do
Apple's faceID is a good example. A new true depth hardware was developed.
    Camera projected 3000 dots on face and also captured infrared images to accurately analyze
Self driving uses RGBD, lidars, radars. CV is used for segmentation at pixel level. Guage human behavior.
Medical CT data, etc used to build models and help doctors
Pose estimation to detect body language
Emotion detection (Media, automotive for state of driver)
Aerial images taken by drones

* Convolution NN
CV aims at understanding pixels. Each pixel is RGB channel. 0-255 is strength of channel.
Grayscale is only one channel. i.e. shade of gray to be potrayed in image
CV should be able to detect the characteristics of image irrespective of linear, rotation, scale transforms
Spatial invariance intuition is once a set of neurons learn about horizontal lines, they can detect it
    anywhere in the image
A simple NN cannot be used since size would be tremendously large given number of pixels and information
    contained within the pixels of the image
CAP - convolution, activation, and pooling
    Conv operation- kernel, stride, padding
    activation- ReLU
    pooling- Max and Avg pooling
Conv takes input activation map -> applies kernel/filter -> output activation map (1 per kernel)
    Kernel is odd number matrix i.e. 3x3 or 5x5
    Kernel detects unique features like lines, edges, eyes, ears, nose, etc
    Each kernel will have same number of channels as incoming image
    Number of outputs is same as number of kernels
A study showed how CovNet learnt. For example:
    layer-1 learnt lines, edges
    layer-2 learnt curvature
    layer-3 learnt facial features
The crux of CovNet is that each layer builds and learns on top of previous

* Strides, Padding, Pooling (Hyperparameters)
Stride determines step size to move kernel left to right and top to bottom
    e.g. stride of 2 will skip a pixel
Padding adds pixels on the boundary to control output activation size
    Value of pixel is 0
    Thickness of padding is padding number
    Sudden decrease in size could also mean loss of features
Along row dimension: output_size = (input_size + 2*padding - kernel size) / stride + 1
Pooling slides a kernel over incoming activation layer to downsample
    Stride for kernel equal to its size
CovNets can handle spatial data like images
Can understand object characteristics under challenging conditions
Memory and computation efficiency

* MNIST
0-9 digits and popular dataset 28x28 grayscale. 60K train and 10K test.
Normalization to transform image so that mean: 0 and stddev: 1
Since every channel has uniform distribution, gradients act uniformly, and NN learns faster.
    Vanishing (low) and exploding (high) gradients are avoided

* Model
- Dropout
Overfitting when model starts to memorize and it cannot generalize well
Variety of techniques fall under regularization
Dropout is a regularization technique
    Each iteration, randomly switch off N% of neurons
    All neurons get a chance to learn training data
    No few neurons keep focusing on the data

SECTION-4
* Learning sequences
Time series or Text
Natural language modeling is determining probability distribution over sequences of words
Applications: 
    Predictive keyboards (learn prob distribution over sequence of words) 
    Machine translation
    Sentiment analysis (also provided as API by clouds)
    Financial forecast
    Named-entity recognition (recognize celebrities in news articles)
    Document processing or understanding natural language

* Word embedding
Represent text numerically as vectors
Vocabulary: Unordered set of unique words representing the problem
One-hot encoding
    Vectors become too large and sparse so not memory efficient
    Similar words wouldn't be grouped together since they are same size in embedding
Word embedding
    Float datatype
    Reduced size words
    learnt from corpus
    denser than one-hot embedding
    each dimension captures a different characteristic (e.g. gender)
    embedding is learnt and then passed on to the RNN
Contextual Similarity
    walking and running | walked and ran
    man and king | queen and woman
torch has Embedding layer that gets placed before RNN layer
As network is trained, embedding layer keeps getting better at contextual similarities of words
Pre-trained industry standard embeddings can also be downloaded to save time and effort (GLoVe and word2vec)
RNN is suitable for temporal reasoning (memory built in)
    Has a feedback loop after 1 time step
    Internal state of cell (h) depends on previous state and current input x
    Three weights: U- weighs input, V- weights previous state, W-weights output
    For each cell, there are batch propagation through time (bptt) RNN states t-1, t, and t+1
    All bptt are changed based on previous three states
Hyperparameters:
    batch size (number of sequences)
    embedding size
    number of neurons in h
    vocabulary size
    batch propagation through time size i.e. how many words in sequence to be sent for training at once
Input Size: batch size X embedding size
Output Size: batch size X vocabulary size (prob of this word occurring)
RNN Process Sequences:
    One-One- Vanilla
    One-Many- Image captioning
    Many-Many- Translation/Video Classification
    Many-One- Sentiment Analysis
Vanishing gradients: Use LSTM, GRU (gated recurring units)
Exploding gradients: gradient clipping
P(w_t | context), t in Vocabulary
"""

run_section_1 = False
run_section_2 = False
run_section_3 = False
run_section_4 = True

import IPython
import os
import torch
import torch.nn
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import torchvision
from torch.autograd import Variable
from torch import FloatTensor
import logging
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import multiprocessing

multiprocessing.set_start_method("fork")

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.warning(f"Torch version: {torch.__version__}")

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)


def main(epochs, train_loader, test_loader, model, criterion, optimizer, device=DEVICE):
    # Move model to the specified device
    model = model.to(device)

    # Epochs
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    for iter in range(epochs):
        train_correct = [0, 0]  # Accurate, Total
        train_batch_loss = []

        test_correct = [0, 0]  # Accurate, Total
        test_batch_loss = []

        # Training iterations
        for idx, (items, classes) in enumerate(train_loader):
            # Move data and labels to the device
            input_data = items.to(device)
            input_classes = classes.to(device)

            model.train()
            optimizer.zero_grad()
            outputs = model(input_data)
            loss = criterion(outputs, input_classes)
            loss.backward()
            optimizer.step()

            # Bookkeeping - Training
            _, predicted = torch.max(outputs.data, 1)
            train_correct[0] += (predicted == input_classes).sum().item()
            train_correct[1] += classes.size(0)
            train_batch_loss.append(loss.item())

        train_loss.append(np.average(train_batch_loss))
        train_accuracy.append(train_correct[0] / train_correct[1] * 100)

        # Testing iterations
        with torch.no_grad():
            for idx, (items, classes) in enumerate(test_loader):
                # Move data and labels to the device
                input_data = items.to(device)
                input_classes = classes.to(device)

                model.eval()
                outputs = model(input_data)
                loss = criterion(outputs, input_classes)

                # Bookkeeping - Testing
                _, predicted = torch.max(outputs.data, 1)
                test_correct[0] += (predicted == input_classes).sum().item()
                test_correct[1] += classes.size(0)
                test_batch_loss.append(loss.item())

        test_loss.append(np.average(test_batch_loss))
        test_accuracy.append(test_correct[0] / test_correct[1] * 100)

        logger.info(
            f"Epoch {iter + 1}/{epochs} | Train Accuracy: {train_accuracy[-1]:.2f}% | Test Accuracy: {test_accuracy[-1]:.2f}% | Train Loss: {train_loss[-1]:.4f} | Test Loss: {test_loss[-1]:.4f}"
        )


# SECTION-1
if run_section_1:
    t = FloatTensor(2, 3)
    logger.info(f"Uninitialized Tensor: {t}")

    # y = 2x
    x = Variable(FloatTensor([11.2]), requires_grad=True)
    y = 2 * x

    y.backward()
    logger.info(f"X grad: {x.grad}")
    logger.info(f"Y grad: {y.grad}")

    tf_composed = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=os.path.join(ROOT, "data"),
        train=True,
        transform=tf_composed,
        download=True,
    )
    # trainloader will have a batch at a time
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=10, shuffle=True, num_workers=2, pin_memory=True
    )

    for i, data in enumerate(trainloader):
        sample, label = data
        logger.info(f"Sample index: {i}")
        logger.info(
            f"Data type: {type(sample)}\n Size: {sample.size()}\n Labels: {label}"
        )
        break


# SECTION-2
if run_section_2:
    # HELPERS

    label_idx = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

    class IrisDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            item = self.data.iloc[index].values
            return (item[0:4].astype(np.float32), item[4].astype(np.int64))

        def __len__(self):
            return self.data.shape[0]

    def get_datasets(iris_file, train_ratio=0.80):
        labels = {"class": label_idx}
        data = pd.read_csv(iris_file)
        data.replace(labels, inplace=True)

        train_df = data.sample(frac=train_ratio, random_state=3)
        test_df = data.loc[~data.index.isin(train_df.index), :]

        return IrisDataset(train_df), IrisDataset(test_df)

    class IrisNet(torch.nn.Module):
        def __init__(self, input_size, hidden_layer1, hidden_layer2, output_size):
            # Output size is the class size
            super(IrisNet, self).__init__()
            self.layer1 = torch.nn.Linear(input_size, hidden_layer1)
            self.relu1 = torch.nn.ReLU()
            self.layer2 = torch.nn.Linear(hidden_layer1, hidden_layer2)
            self.relu2 = torch.nn.ReLU()
            self.layer3 = torch.nn.Linear(hidden_layer2, output_size)

        def forward(self, x):
            # x will be number of data instances in a batch
            out = self.layer1(x)
            out = self.relu1(out)
            out = self.layer2(out)
            out = self.relu2(out)
            out = self.layer3(out)
            return out

    train_data, test_data = get_datasets(os.path.join(ROOT, "data", "iris_data.txt"))
    logger.info(f"Instances in train data: {len(train_data)}")
    logger.info(f"Instances in test data: {len(test_data)}")
    model = IrisNet(4, 100, 50, 3)
    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.SGD(
        model.parameters(), learning_rate, nesterov=True, momentum=0.9, dampening=0
    )

    iris_train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=60, shuffle=True, num_workers=2, pin_memory=True
    )
    iris_test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=60, shuffle=True, num_workers=2, pin_memory=True
    )
    main(500, iris_train_loader, iris_test_loader, model, criterion, optimizer)


# SECTION-3
if run_section_3:
    tf_composed = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )
    train_data = torchvision.datasets.MNIST(
        root=os.path.join(ROOT, "data"),
        train=True,
        download=True,
        transform=tf_composed,
    )
    test_data = torchvision.datasets.MNIST(
        root=os.path.join(ROOT, "data"),
        train=False,
        download=True,
        transform=tf_composed,
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, shuffle=True, batch_size=1024, num_workers=2, pin_memory=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, shuffle=True, batch_size=1024, num_workers=2, pin_memory=True
    )

    class MNISTNet(torch.nn.Module):
        def __init__(self):
            super(MNISTNet, self).__init__()
            # Input size 28x28x1
            self._conv1 = torch.nn.Conv2d(1, 20, kernel_size=5)
            # Input size 24x24x20
            self._relu1 = torch.nn.ReLU()

            self._conv2 = torch.nn.Conv2d(20, 30, kernel_size=5)
            # Input size 20x20x30
            self._conv2_dropout = torch.nn.Dropout2d(p=0.5)
            # Input size 20x20x30
            self._pool1 = torch.nn.MaxPool2d(kernel_size=2)
            # Input size 10x10x30
            self._relu2 = torch.nn.ReLU()

            self._layer1 = torch.nn.Linear(3000, 500)
            self._layer2 = torch.nn.Linear(500, 10)

        def forward(self, x):
            x = self._conv1(x)
            x = self._relu1(x)
            x = self._conv2(x)
            x = self._conv2_dropout(x)
            x = self._pool1(x)
            x = self._relu2(x)
            x = x.view(-1, 3000)
            x = self._layer1(x)
            x = F.relu(x)
            x = F.dropout(x, training=True)
            x = self._layer2(x)

            return x

    model = MNISTNet()
    main(
        100,
        mnist_train_loader,
        mnist_test_loader,
        model,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(
            model.parameters(), lr=0.01, nesterov=True, momentum=0.9, dampening=0
        ),
    )


# SECTION-4
if run_section_4:

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

        return total_loss[0] / len(data_source)

    def evaluate(data_source, bs_valid, bptt_size, vocab_size):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0
        hidden = model.init_hidden(bs_valid)

        for i in range(0, data_source.size(0) - 1, bptt_size):
            data, targets = get_batch(data_source, i, evaluation=True)

            output, hidden = model(data, hidden)
            output_flat = output.view(-1, vocab_size)

            total_loss += len(data) * criterion(output_flat, targets).data
            hidden = Variable(hidden.data)

        return total_loss[0] / len(data_source)

    best_val_loss = None

    def run(epochs, lr, bptt_size, bs_train, bs_valid, vocab_size):
        global best_val_loss

        for epoch in range(0, epochs):
            train_loss = train(train_data, lr, bptt_size, bs_train)
            val_loss = evaluate(val_data, bs_valid, bptt_size, vocab_size)
            logger.info(f"Train Loss: {train_loss}. Valid Loss: {val_loss}")

            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "../model/rnn.model.pth")

    run(5, 0.001, bptt_size, bs_train, bs_valid, vocab_size)

    IPython.embed()
