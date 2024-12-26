#! /usr/bin/python3

import torch
from torch.autograd import Variable
import logging
from IPython import embed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

x = Variable(
    data=torch.randn(size=(10,), device="cpu", dtype=torch.float32), requires_grad=True
)
embed()
