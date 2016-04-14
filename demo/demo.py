from __future__ import print_function

import sys
import os
import time
import thread

import numpy as np

import theano
import theano.tensor as T
import lasagne

import flask
from flask import render_template, Response, jsonify



# Hold options as static element in the opts class
class opts():
    hidden_size = 50
    seq_len = 25         # Data sequence length
    gradient_steps = 20  # Truncated BPTT length
    data_offset = 15     # Offset for every new input sequence
    batch_size = 50
    n_epochs = 500
    lr = 0.1


def build_rnn_1(input_var, dim):
    # ----- INPUT LAYER -----
    l_in = lasagne.layers.InputLayer(shape=(None, opts.seq_len, dim), input_var=input_var)
    
    io_gate_parameters = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.))
    
    forget_gate_parameters = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(5.))
    
    cell_parameters = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        W_cell=None, b=lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.tanh)
    
    # ----- LSTM LAYER -----
    l_lstm = lasagne.layers.recurrent.LSTMLayer(
        l_in, opts.hidden_size,
        ingate=io_gate_parameters, forgetgate=forget_gate_parameters,
        cell=cell_parameters, outgate=io_gate_parameters,
        learn_init=True, grad_clipping=50., gradient_steps=opts.gradient_steps)
    
    # ----- FC LAYER -----
    batch_size, _, _ = l_in.input_var.shape
    l_reshape = lasagne.layers.ReshapeLayer(l_lstm, (batch_size * opts.seq_len, opts.hidden_size))
    l_dense = lasagne.layers.DenseLayer(
        l_reshape, num_units=dim, nonlinearity=lasagne.nonlinearities.softmax)
    
    return l_dense


# Define function to get batches of preprocessed data.
def get_batch(b):
    if (b+1)*opts.batch_size*opts.data_offset - opts.data_offset + opts.seq_len + 1 >= len(data):
        return None, None
    X = np.zeros((opts.batch_size, opts.seq_len, vocab_size), dtype=theano.config.floatX)
    y = np.zeros((opts.batch_size, opts.seq_len, vocab_size), dtype=np.int8)
    
    for i in xrange(opts.batch_size):
        c = b*opts.data_offset*opts.batch_size + opts.data_offset*i
        for j in xrange(opts.seq_len):
            X[i, j, char_to_ix[data[c]]] = 1.0
            y[i, j, char_to_ix[data[c+1]]] = 1.0
            c += 1
    
    return X, y.reshape((opts.batch_size*opts.seq_len, vocab_size))


def sample_text(length=200):
    # First take a random piece of bootstrap text
    start = np.random.randint(0, len(data)-opts.seq_len)
    s = data[start:start+opts.seq_len]
    
    # Convert to proper input data shape (here, batch size = 1)
    s_np = np.zeros((1, opts.seq_len, vocab_size), dtype=theano.config.floatX)
    for i in xrange(opts.seq_len):
        s_np[0, i, char_to_ix[s[i]]] = 1.0
    
    # Start sampling loop
    res = ''
    for k in xrange(length):
        # Predict next character
        predict = sample(s_np)
        predict_i = np.random.choice(range(vocab_size), p=predict.ravel())
        res += ix_to_char[predict_i]
        
        # Update s_np
        s_np[0, 0:-1, :] = s_np[0, 1:, :]
        s_np[0, -1, :] = 0.0
        s_np[0, -1, predict_i] = 1.0
    
    return res
    

data = open('../tinyshakespeare.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('Vocabulary size = ' + str(vocab_size) + '; total data size = ' + str(data_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

input_var = T.tensor3('inputs')
output_var = T.bmatrix('outputs')

global network
network = build_rnn_1(input_var, vocab_size)

network_output = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(network_output, output_var).mean()
all_params = lasagne.layers.get_all_params(network)
updates = lasagne.updates.adam(loss, all_params)

global train
train = theano.function(
    [input_var, output_var],
    loss, updates=updates)

global sample
sample = theano.function(
    [input_var], network_output[-1,:])

global output_text
output_text = sample_text(300)

print(output_text)

def start_training():
    global output_text
    for epoch in range(opts.n_epochs):
        b = 0.0
        counter = 0
        while True:
            X, y = get_batch(int(b))
            if X is None or y is None:
                break
            train(X, y)
            if counter % 50 == 1:
                output_text = sample_text(300)
                print("-- UPDATE SAMPLE")
            b += 1.0
            counter += 1

thread.start_new_thread(start_training, ())

app = flask.Flask(__name__)

@app.route('/sample')
def sample_page():
    global output_text
    print("AJAX REQUEST")
    return jsonify(result=output_text)

@app.route('/home')
def home():
    return render_template('training.html')


if __name__ == '__main__':
    app.run()