#!/usr/bin/env python
# coding=utf-8
import numpy as np
import random

class dataProcessor:

    def __init__(self,trainingFile,testingFile):
        self.trainingFile = trainingFile #存储训练数据的文件名
        self.testingFile = testingFile #存储测试数据的文件名
        #self.path = r'/home/gqy/Desktop/AnnPredictRpProbability/Data/'
        #self.path = r'/home/lirui/Projects/DeepLearning/GQY/GQY-20170225-AnnPositioning/AnnPositioning/Data/' #训练文件和测试文件的路径
        #self.path = r'/Users/tmac/Desktop/AnnPredictRpProbability/Data/'
        self.path = r'/home/gqy/Desktop/Experiment/AnnPredictRpProbability/Data/'
        self.apList = self.__getApList() #所有出现的AP的列表，即特征列表
        self.traningApFingerprints,self.trainingCoordinates = self.__getApFingerprintsAndCoordinates(self.trainingFile)
        self.testingApFingerprints,self.testingCoordinates = self.__getApFingerprintsAndCoordinates(self.testingFile)
        self.originalTrainingApFingerprints,self.originalTrainingCoordinates = self.__getApFingerprintsAndCoordinates(self.trainingFile,isShuffle=False)
        self.originalTestingApFingerprints,self.originalTestingCoordinates = self.__getApFingerprintsAndCoordinates(self.testingFile,isShuffle=False)
        '''
        返回指纹数据的基本信息apRssiInfoWithCorresponing_Coordinates，
        格式如下包括多个位置（如果是训练集数据，则每个位置包含100组数据）
        [
          [ [Rssi_1,Rssi_2, ... ,Rssi_L],[x1,y1] ]
          [ [Rssi_1,Rssi_2, ... ,Rssi_L],[x1,y1] ]
          ...100个
          [ [Rssi_1,Rssi_2, ... ,Rssi_L],[x1,y1] ]
          [ [Rssi_1,Rssi_2, ... ,Rssi_L],[x2,y2] ]
          [ [Rssi_1,Rssi_2, ... ,Rssi_L],[x2,y2] ]
          ...100个
          [ [Rssi_1,Rssi_2, ... ,Rssi_L],[x2,y2] ]
          ...
          [ [Rssi_1,Rssi_2, ... ,Rssi_L],[xn,yn] ] ]
          '''
    def __apRssiInfoWithCorresponing_Coordinates(self,filename,isShuffle):
        apRssiInfoWithCorresponing_Coordinates = []
        fingerprint = []
        apRssiList = []
        apMac = ''
        apRssi = ''
        xAxis = ''
        yAxis = ''
        macIndex = -1000

        f = open(self.path+filename,'r')
        for line in f.readlines():
            if line.startswith('X_Axis'):
                xAxis = line[7:-1]
                continue
            if line.startswith('Y_Axis'):
                yAxis = line[7:-1]
                apRssiList = [-100.0]*len(self.apList)
                fingerprint.append(apRssiList)
                fingerprint.append([float(xAxis),float(yAxis)])
                continue
            if line.startswith('BSSID'):
    	        apMac= line[6:-1]
                macIndex = self.apList.index(apMac)
    	        continue
    	    if line.startswith('SignalStrength'):
    	        apRssi=line[15:]
                apRssiList[macIndex] = float(apRssi)
                continue
            if line.startswith('*'):
                apRssiInfoWithCorresponing_Coordinates.append(fingerprint)
                fingerprint = []
        f.close()
        
        #return np.load("apRssiInfoWithCorresponing_Coordinates.npy").tolist()
        apRssiInfoWithCorresponing_Coordinates = np.array(apRssiInfoWithCorresponing_Coordinates)
        if isShuffle:
            #将原始数据打乱顺序，以便后续的训练
            np.random.shuffle(apRssiInfoWithCorresponing_Coordinates)
            np.save("apRssiInfoWithCorresponing_Coordinates.npy", apRssiInfoWithCorresponing_Coordinates)
        return apRssiInfoWithCorresponing_Coordinates.tolist()


    #返回指纹中出现的所有AP的Mac的地址的列表
    def __getApList(self):
        ApSet = set([])
        for filename in [ self.trainingFile, self.testingFile ]:
            f = open(self.path+filename,'r')
            for line in f.readlines():
                if line.startswith('BSSID'):
                    ApMac= line[6:-1]
                    ApSet = ApSet | set([ApMac])
            f.close()
        return list(ApSet)

    #返回左右出现的RP的坐标构成的列表
    def __getCoordinatesList(self):
        CoordinatesSet = set([])
        xAxis=''
        yAxis=''
        for filename in [ self.trainingFile, self.testingFile ]:
        #for filename in [ self.trainingFile]:
            f = open(self.path+filename,'r')
            for line in f.readlines():
                if line.startswith('X_Axis'):
                    xAxis = line[7:-1]
                if line.startswith('Y_Axis'):
                    yAxis = line[7:-1]
                    CoordinatesSet = CoordinatesSet | set([(float(xAxis),float(yAxis))])
            f.close()
        #print len(list(CoordinatesSet))
        #print list(CoordinatesSet)
        return list(CoordinatesSet)

    '''
    两个返回值
    第一个：返回所有RP位置上所有AP的Rssi特征,即指纹
    第二个：返回该AP特征对应的RP的坐标，相当与监督学习中的labels
    '''
    def __getApFingerprintsAndCoordinates(self,filename,isShuffle=True):
        apFingerprints = []
        correspondingCoordinates = []
        for fingerprint in self.__apRssiInfoWithCorresponing_Coordinates(filename,isShuffle):
            apFingerprints.append(fingerprint[0])
            correspondingCoordinates.append(np.array(fingerprint[1]))
        return apFingerprints,correspondingCoordinates

    #数据的归一化函数min-max normalization和z-score normalization
    #type参数可选'min-max'或'z-score'
    def __normalization(self,apFingerprints,type='min-max'):
        normalizedApFingerprints = np.array(apFingerprints)
        #使用min-max归一化
        if type == 'min-max':
            #每个AP的信号强度最大值和最小值
            max_Rss = np.max(self.traningApFingerprints,axis=0)
            min_Rss = np.min(self.traningApFingerprints,axis=0)
            #print '最大Rss为：'
            #print max_Rss
            #print '最小Rss为：'
            #print min_Rss
            for j in range(np.array(apFingerprints).shape[1]):
                for i in range(np.array(apFingerprints).shape[0]):
                    if max_Rss[j]>min_Rss[j]:
                        normalizedApFingerprints[i,j]=(normalizedApFingerprints[i,j]-min_Rss[j]+0.0)/(max_Rss[j]-min_Rss[j])
                    else:
                        normalizedApFingerprints[i,j] = 0.0
        #使用z-score归一化
        elif type == 'z-score':
            #每个AP的信号强度均值和标准差
            mean_Rss = np.mean(self.traningApFingerprints,axis=0)
            std_Rss =np.std(self.traningApFingerprints,axis=0)
            for j in range(apFingerprints.shape[1]):
                for i in range(apFingerprints.shape[0]):
                    normalizedApFingerprints[i,j]=(normalizedApFingerprints[i,j]-mean_Rss[j]+0.0)/std_Rss[j]
        return normalizedApFingerprints



    #获取训练数据中的AP的RSS指纹
    def getTrainingApFingerprints(self):
        return np.array(self.traningApFingerprints)
        #return self.__normalization(self.traningApFingerprints)
    #获取训练数据中的指纹对应的RP的位置坐标
    def getTrainingCoordinates(self):
        return np.array(self.trainingCoordinates)
    #获取测试数据中的AP的RSS指纹
    def getTestingApFingerprints(self):
        return np.array(self.testingApFingerprints)
        #return self.__normalization(self.testingApFingerprints)
    #获取测试数据中的指纹对应的RP的真实位置坐标
    def getTestingCoordinates(self):
        return np.array(self.testingCoordinates)

    #获取训练数据中的指纹对应的RP的位置坐标的编号所构成的列表
    def getTrainingCoordinatesId(self):
        coordinatesList = self.__getCoordinatesList()
        trainingCoordinatesId = np.zeros([len(self.trainingCoordinates),len(coordinatesList)])
        for i,coordinate in  enumerate(self.trainingCoordinates):
            coordinate = tuple(coordinate)
            j = coordinatesList.index(coordinate)
            trainingCoordinatesId[i,j] = 1.0
        return trainingCoordinatesId

    #获取测试数据中的指纹对应的RP的位置坐标的编号所构成的矩阵．矩阵的每一行都是一个one-hot vector
    def getTestingCoordinatesId(self):
        coordinatesList = self.__getCoordinatesList()
        testingCoordinatesId = np.zeros([len(self.testingCoordinates),len(coordinatesList)])
        for i,coordinate in  enumerate(self.testingCoordinates):
            coordinate = tuple(coordinate)
            j = coordinatesList.index(coordinate)
            testingCoordinatesId[i,j] = 1.0
        return testingCoordinatesId

    #获取训练数据中的AP的time averaged指纹和对应的指纹
    def getTimeAveragedTrainingApFingerprintsAndCoordinates(self):
        timeAveragedTrainingApFingerprints = []
        trainingCoordinates = []

        total = np.zeros(len(self.apList))
        count = 0
        for index in range(len(self.traningApFingerprints)):
            total += np.array(self.traningApFingerprints[index])
            count += 1
            if (index < (len(self.traningApFingerprints)-1) and self.trainingCoordinates[index] != self.trainingCoordinates[index+1]) or index == (len(self.traningApFingerprints)-1):
                timeAveragedTrainingApFingerprint = total / count
                timeAveragedTrainingApFingerprints.append(timeAveragedTrainingApFingerprint.tolist())
                trainingCoordinates.append(self.trainingCoordinates[index])
                total = np.zeros(len(self.apList))
                count = 0

        return np.array(timeAveragedTrainingApFingerprints),np.array(trainingCoordinates)

    #随机从测试数据中选择num条数据，用于测试模型的性能
    #返回随机挑选出的Ap指纹和对应的RP位置坐标
    def getTestingSamplesRandomly(self,num):
        testingApFingerprints = []
        testingCorrespondingCoordinates = []
        testingRssiInfoWithCorrespondingCoordinates = self.__apRssiInfoWithCorresponing_Coordinates(self.testingFile)
        #random.shuffle(testingRssiInfoWithCorrespondingCoordinates)
        for fingerprint in testingRssiInfoWithCorrespondingCoordinates[:num]:
            testingApFingerprints.append(fingerprint[0])
            testingCorrespondingCoordinates.append(fingerprint[1])

        return self.__normalization(testingApFingerprints),np.array(testingCorrespondingCoordinates)

if __name__ == '__main__':
    dp = dataProcessor(r'data4trainingNexus',r'bai4testing_1_4.log')
    #dp.getTimeAveragedTrainingApFingerprintsAndCoordinates()
    #dp.getTestingSamplesRandomly(15)
    #print '原始trainingApFingerprints:'
    #print dp.traningApFingerprints[:20]
    #print ' 归一化后的traingingApFingerprints:'
    #print dp.getTrainingApFingerprints()[:20]

    #dp.__getCoordinatesList()
    train = dp.getTrainingCoordinatesId()
    #print '*'*30
    test = dp.getTestingCoordinatesId()
    print 'train.shape:',train.shape
    print 'test.shape:',test.shape
    reply1 = 0
    reply2 = 0
    for i in range(train.shape[0]):
        if 1 in train[i,:] and np.sum(train[i,:])==1:
            reply1 +=1
    print 'reply1:',reply1

    for i in range(test.shape[0]):
        if 1 in test[i,:] and np.sum(test[i,:])==1:
            reply2 += 1
    print 'reply2:',reply2
