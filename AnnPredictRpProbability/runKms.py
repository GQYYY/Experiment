#-*- coding:utf-8 -*-

import dataProcessor as dp
from kmeans import KMeans
import numpy as np
if __name__ == '__main__':
    dataProcessor = dp.dataProcessor(r'data4trainingNexus',r'bai4testing_1_4.log')
    #获得训练指纹(12400*92)
    trainingApFingerprints = dataProcessor.getTrainingApFingerprints() +100
    #获取和训练指纹所对应的RP的编号(12400*124)
    trainingCoordinatesId = dataProcessor.getTrainingCoordinatesId()
    #获取测试指纹
    testingApFingerprints = dataProcessor.getTestingApFingerprints() +100
    #获取和训练指纹所对应的RP的编号
    testingCoordinatesId = dataProcessor.getTestingCoordinatesId()

    kms = KMeans(k=20)
    kms.train(trainingApFingerprints)
    np.save('centers.npy', kms._centers)
    np.save('labels.npy', kms._label)

