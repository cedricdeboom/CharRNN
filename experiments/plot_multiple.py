"""
Create plots of multiple flavours
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif', serif='Times', style='normal', size=16)


def get_data_from_file(filename):
    x = []
    y = []
    with open(filename, 'r') as f:
        for line in f:
            segments = line.split(':')
            if segments[0].startswith('Mean'):
                # m = float(segments[1])
                # for i in xrange(len(x)):
                #     x[i] *= m
                break
            x.append(int(segments[0])*50)
            y.append(float(segments[1]))
    return x, y

flavours = ['flavour_1', 'flavour_2', 'flavour_3', 'flavour_4']
colours = ['b-', 'r-', 'g-', 'y-']

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)

for i, fl in enumerate(flavours):
    for filename in os.listdir(fl):
        if filename.endswith('.txt'):
            x, y = get_data_from_file(os.path.join(fl, filename))
            plt.semilogx(x, y, colours[i], alpha=0.2)

ax.set_xlim([50, 500001])
ax.set_xlabel('Number of input train sequences')
ax.set_ylabel('Perplexity')

plt.savefig('multiple-batch.pdf')
plt.close()