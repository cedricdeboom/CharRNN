from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import lasagne

import LSTMLayer_v2

#######
# Flavour 1 - seq2seq training, seq2sample prediction
#######

if len(sys.argv) < 3:
    print('Not enough args.')
    sys.exit(0)

seed = np.random.seed(int(sys.argv[1]))
configs = int(sys.argv[2])

data = open('../tinyshakespeare.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('Vocabulary size = ' + str(vocab_size) + '; total data size = ' + str(data_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# Hold options as static element in the opts class
class opts():
    hidden_size = 50
    seq_len = 25         # Data sequence length
    gradient_steps = 25  # Truncated BPTT length
    data_offset = 15     # Offset for every new input sequence
    batch_size = 50


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


def get_batch(data, b, b_size, seq_len, offset):
    start = int(float(len(data))/float(b_size))
    if start*(b_size - 1) + offset*b + seq_len >= len(data):
        return None, None
    X = np.zeros((b_size, seq_len, vocab_size), dtype=theano.config.floatX)
    y = np.zeros((b_size, seq_len, vocab_size), dtype=np.int8)

    for i in xrange(b_size):
        c = start*i + offset*b
        for j in xrange(seq_len):
            X[i, j, char_to_ix[data[c]]] = 1.0
            y[i, j, char_to_ix[data[c+1]]] = 1.0
            c += 1

    return X, y.reshape((b_size*seq_len, vocab_size))

def raw_perplexity(input_data, input_targets):
    # Assume input_data.shape = (1, seq_len, vocab_size)
    # Assume input_targets.shape = (seq_len, vocab_size)

    sample_output = sample(input_data)
    p = np.sum(sample_output * input_targets[-1])
    num = -np.log(p)

    return num, 1


train_portion = 0.99
train_data = data[0:int(train_portion*len(data))]
test_data = data[int(train_portion*len(data)):]

pre_params = None

for c in xrange(configs):
    # Generate random params
    opts.seq_len = np.random.randint(5, 51)
    opts.gradient_steps = opts.seq_len
    opts.data_offset = np.random.randint(5, opts.seq_len+1)

    # Create network and compiling functions
    print('Network creation and function compiling...')
    input_var = T.tensor3('inputs')
    output_var = T.bmatrix('outputs')

    network = build_rnn_1(input_var, vocab_size)
    network_output = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(network_output, output_var).mean()
    all_params = lasagne.layers.get_all_params(network)
    updates = lasagne.updates.adam(loss, all_params)

    train = theano.function(
        [input_var, output_var],
        loss, updates=updates)

    sample = theano.function(
        [input_var], network_output[-1, :])

    # Save pre params (initially for first network created)
    if c == 0:
        pre_params = lasagne.layers.get_all_param_values(network, trainable=True)

    # Train procedure
    print("Start training RNN...")
    mean_batch_time = 0.0
    std_batch_time = 0.0
    counter = 0
    printouts = [1, 3, 6, 13, 26, 51, 101, 201, 401, 801, 1601, 3201, 6401, 10001]
    max_counter = max(printouts)

    # Set pre params
    lasagne.layers.set_all_param_values(network, pre_params, trainable=True)

    with open('flavour_1/' + str(opts.data_offset) + '-' + str(opts.seq_len) + '-' + str(opts.gradient_steps) + '.txt', 'w') as fi:
        while True:
            cost = 0.0
            b = 0.0
            while True:
                X, y = get_batch(train_data, int(b), opts.batch_size, opts.seq_len, opts.data_offset)
                if X is None or y is None:
                    break
                counter += 1
                before = time.clock()
                cost += train(X, y)
                after = time.clock()
                delta = float((after - before) - mean_batch_time)
                mean_batch_time += delta/counter
                std_batch_time += delta*((after - before) - mean_batch_time)

                b += 1.0

                if counter in printouts:
                    # Perplexity calculation
                    num, den = 0.0, 0.0
                    tb = 0
                    while True:
                        Xt, yt = get_batch(test_data, tb, 1, opts.seq_len, 1)
                        if Xt is None or yt is None:
                            break
                        n2, d2 = raw_perplexity(Xt, yt)
                        num += n2
                        den += d2
                        tb += 1
                    print(str(counter) + ':' + str(np.exp(num / den)))
                    fi.write(str(counter) + ':' + str(np.exp(num / den)) + '\n')

                    if counter >= max_counter:
                        break

            if counter >= max_counter:
                break

        print('Mean:' + str(mean_batch_time))
        fi.write('Mean:' + str(mean_batch_time) + '\n')
        print('Std:' + str(std_batch_time/(counter-1)))
        fi.write('Std:' + str(std_batch_time/(counter-1)) + '\n')