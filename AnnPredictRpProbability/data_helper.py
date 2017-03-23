#!/usr/bin/env python
# encoding: utf-8

import re
import numpy as np

reg = re.compile(r'\b\w+\b')


def batch_iter(x, y, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(x)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        np.random.seed(10)
        shuffle_indices = np.random.permutation(len(y))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]
    else:
        x_shuffled = x
        y_shuffled = y
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield (x_shuffled[start_index:end_index], y_shuffled[start_index:end_index])
