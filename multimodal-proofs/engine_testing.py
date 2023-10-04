import os
import json
import numpy as np
from engine import Engine
from dataset import Dataset

data = Dataset()
data.set_data('val')

e = Engine()

inds = [5]

x = data.load_qa_pair(inds[0])

e.set_clip(x[0], x[1])

