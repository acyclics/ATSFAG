import tensorflow as tf 
from tensorflow.python.ops.rnn_cell import RNNCell

class lstmCell(RNNCell):
    def __init__(self, num_units, forget_bias=1.0, input_size=None):
        
