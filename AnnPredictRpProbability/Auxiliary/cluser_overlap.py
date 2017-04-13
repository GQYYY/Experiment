#!/usr/bin/env python
# coding=utf-8
from __future__ import division
import numpy as np
import itertools

"""
===========================================================
Obtain the overlapping ratio of RP clusters. Compute the 
overlapping ratio for every pair of clusters.
===========================================================
"""

clst_labels = np.load('../Data_Statistics/clst_labels.npy')
clsts = [np.load('../Data_Statistics/cluster_%d.npy'%clst_label) for clst_label in clst_labels]
print type(clst_labels)
#Compute the overlapping ratio of two clusters, given the cluster pair.
# overlapping_rate = intersection/union
def overlap(clst_label_1,clst_label_2):
    x = clsts[np.where(clst_labels==clst_label_1)[0][0]]
    y = clsts[np.where(clst_labels==clst_label_2)[0][0]]
    x = [tuple(item) for item in x]
    y = [tuple(item) for item in y]
    #intersection
    intersection = list(set(x).intersection(y))
    #union
    union = list(set(x).union(set(y)))

    overlapping_rate = len(intersection)/len(union)
    return overlapping_rate

#overlapping_rate statistics for all cluster pairs
def overlap_stat(threshold=0.3):
    clst_combinations = list(itertools.combinations(clst_labels,2))
    for clst_comb_item in clst_combinations:
        overlapping_rate = overlap(clst_comb_item[0],clst_comb_item[1])
        if overlapping_rate >= threshold:
           print ('cluster ',clst_comb_item,' overlapping_rate: ',overlapping_rate)
    
        #print ('cluster ',clst_comb_item,' overlapping_rate: ',overlapping_rate)


overlap_stat()
