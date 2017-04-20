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

def cluster_anls(cluster_labels,fingerprints,coords_id,coords_list):
    """
    Analyse the cluster each RP belongs to, and the set of RPs each cluster consist of.
    """
    import xlwt
    workbook = xlwt.Workbook()
    '''Analyse the cluster each RP belongs to'''
    sheet1 = workbook.add_sheet(u'sheet_1',cell_overwrite_ok=True)
    sheet1.write(0,0,'RP Coordinate')
    sheet1.write(0,1,'Cluster labels')

    uniq_coords_id = np.unique(coords_id)
    i = 0
    for coord_id in uniq_coords_id:
        i+=1
        sheet1.write(i,0,str(tuple(coords_list.tolist()[coord_id])))
        sheet1.write(i,1,str(np.unique(\
            cluster_labels[np.where(coords_id == coord_id)]\
            .tolist())))

    '''Analyse the set of RPs each cluster contains'''
    sheet2 = workbook.add_sheet(u'sheet_2',cell_overwrite_ok=True)
    sheet2.write(0,0,'Cluster')
    sheet2.write(0,1,'Set of coordinates')

    uniq_clst_labels = np.unique(cluster_labels)
    start = 1
    end = 1
    for clst_label in uniq_clst_labels:
        coord_set = coords_list[coords_id[np.where(cluster_labels == clst_label)],:].tolist()
        coord_set = list(set([tuple(coord) for coord in coord_set]))
        for j in range(len(coord_set)):
            end = start+j
            sheet2.write(end,1,str(coord_set[j]))
        sheet2.write_merge(start,end,0,0,str(clst_label))
        start = end+1
    workbook.save('./Data_Statistics/Rp_Cluster_Relation/rp_cluster_analysis.xls')
    print ('统计信息生成完毕！')



