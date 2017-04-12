#!/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import xlrd 

"""
==========================================================================
Observe the clustering distribution of RPs.
The set of RPs are scattered into the map with different colors, according 
to cluster they belong to.
==========================================================================
"""
workbook = xlrd.open_workbook('/home/gqy/Desktop/Experiment/AnnPredictRpProbability'+
                               '/Data_Statistics/rp_cluster_analysis.xls',\
                               formatting_info=True)
sheet2 = workbook.sheet_by_name('sheet_2')
clst_label_lst = []
for merged_cell in sheet2.merged_cells:
    clst_label = int(sheet2.cell_value(merged_cell[0],merged_cell[2]).encode('utf-8'))
    clst_label_lst.append(clst_label)
    clst = [] #the list of rp coordinates that belong to certain cluster 
    for row in range(merged_cell[0],merged_cell[1]):
        rp_coord = sheet2.cell_value(row,1).encode('utf-8')
        x = float(rp_coord.strip().split(',')[0][1:])
        y = float(rp_coord.strip().split(',')[1][:-1])
        rp_coord = (x,y)
        clst.append(rp_coord)
    np.save('./Data_Statistics/cluster_%d'%clst_label,np.array(clst))
clst_label_lst = np.array(clst_label_lst)
np.save('./Data_Statistics/clst_labels',clst_label_lst)

plt.figure()
colors = ['#000000','#ccad60','#bff128','#0a481e','#49759c',
           '#fb5ffc','#e50000','#33b864','#490648','#fcc006',
           '#a8a495','#6c3461','#f8481c','#a2bffe','#0343df']
markers = ['+','x','o','v','^','<','>','s','p','h','D','d','_','|','2']
scalars = [100,100,150,10,10,10,10,15,15,15,15,18,80,80,50]

for clst_label,color,marker,scalar in zip(clst_label_lst,colors,markers,scalars):
    rp_clst = np.load('./Data_Statistics/cluster_%d.npy'%clst_label)
    plt.scatter(rp_clst[:,0],rp_clst[:,1],c=color,marker=marker,s=scalar,label=str(clst_label))

#x = np.arange(15.4,18,0.8)
#y = np.arange(0,52.0,0.8)
#X,Y = np.meshgrid(x,y)
#X = X.ravel()
#Y = Y.ravel()
#plt.scatter(X,Y,marker=',')

plt.axis([10,24,-10,60])
plt.legend(loc='best')
plt.show()


