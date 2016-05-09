"""
Create plots of one flavour
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif', serif='Times', style='normal')


def get_data_from_file(filename):
    x = []
    y = []
    with open(filename, 'r') as f:
        for line in f:
            segments = line.split(':')
            if segments[0].startswith('Mean'):
                m = float(segments[1])
                for i in xrange(len(x)):
                    x[i] *= m
                break
            x.append(int(segments[0]))
            y.append(float(segments[1]))
    return x, y

flavour = 'flavour_1'

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)

for filename in os.listdir(flavour):
    if filename.endswith('.txt'):
        x, y = get_data_from_file(os.path.join(flavour, filename))
        plt.semilogx(x, y, 'b-')

ax.set_xlabel('Number of training batches')
ax.set_ylabel('Perplexity')

plt.savefig(flavour + '.pdf')
plt.close()