#!/usr/bin/env python
# coding=utf-8
import numpy as np

class dataProcessor:

    def __init__(self,trainingFile,testingFile):
        self.trainingFile = trainingFile #存储训练数据的文件名
        self.testingFile = testingFile #存储测试数据的文件名
        #self.path = r'/home/gqy/Desktop/AnnPredictRpProbability/Data/'
        #self.path = r'/home/lirui/Projects/DeepLearning/GQY/GQY-20170225-AnnPositioning/AnnPositioning/Data/' #训练文件和测试文件的路径
        #self.path = r'/Users/tmac/Desktop/AnnPredictRpProbability/Data/'
        self.path = r'./Data/'
        self.apList = self.__getApList() #所有出现的AP的列表，即特征列表
        self.coordList = self.__getCoordinatesList() #训练集和测试集中所有出现的RP的位置坐标列表
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

    #获取训练数据中的AP的RSS指纹
    def getTrainingApFingerprints(self):
        return np.array(self.traningApFingerprints)
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

    #获取原始(未经shuffle)训练数据中的AP的RSS指纹
    def getOriginalTrainingApFingerprints(self):
        return np.array(self.originalTrainingApFingerprints)
    #获取原始训练数据中的指纹对应的RP位置坐标
    def getOriginalTrainingCoordinates(self):
        return np.array(self.originalTrainingCoordinates)
    #获取原始(未经shuff)测试数据中的AP的RSS指纹
    def getOriginalTestingApFingerprints(self):
        return np.array(self.originalTestingApFingerprints)
    #获取原始测试数据中的RP的真是坐标位置
    def getOriginalTestingCoordinates(self):
        return np.array(self.originalTestingCoordinates)

    #获取AP列表
    def getApList(self):
        return self.apList

    #获取RP位置坐标列表
    def getCoordinatesList(self):
        return self.coordList

    #获取数据中的指纹对应的RP的位置坐标的编号所构成的列表
    def getCoordinatesId(self,coordinates):
        coordinatesList = self.coordList
        coordinatesId = []
        for coord in coordinates:
            coord = tuple(coord)
            coordinatesId.append(coordinatesList.index(coord))
        return np.array(coordinatesId)


if __name__ == '__main__':
    dataProcessor = dataProcessor(r'data4trainingNexus',r'bai4testing_1_4.log')
    trainingApFingerprints = dataProcessor.getTrainingApFingerprints()
    trainingCoordinates = dataProcessor.getTrainingCoordinates()
    trainingCoordinatesId = dataProcessor.getCoordinatesId(trainingCoordinates)

    originalTrainingApFingerprints = dataProcessor.getOriginalTrainingApFingerprints()
    originalTrainingCoordinates = dataProcessor.getOriginalTrainingCoordinates()
    originalTrainingCoordinatesId = dataProcessor.getCoordinatesId(originalTrainingCoordinates)

    testingApFingerprints = dataProcessor.getTestingApFingerprints()
    testingCoordinates = dataProcessor.getTestingCoordinates()
    testingCoordinatesId = dataProcessor.getCoordinatesId(testingCoordinates)

    originalTestingApFingerprints = dataProcessor.getOriginalTestingApFingerprints()
    originalTestingCoordinates = dataProcessor.getOriginalTestingCoordinates()
    originalTestingCoordinatesId = dataProcessor.getCoordinatesId(originalTestingCoordinates)

    coordinatesList = dataProcessor.getCoordinatesList()
    apList = dataProcessor.getApList()

    np.save('./Data/trainingApFingerprints',trainingApFingerprints)
    np.save('./Data/trainingCoordinates',trainingCoordinates)
    np.save('./Data/trainingCoordinatesId',trainingCoordinatesId)
    np.save('./Data/Original/originalTrainingApFingerprints',originalTrainingApFingerprints)
    np.save('./Data/Original/originalTrainingCoordinates',originalTrainingCoordinates)
    np.save('./Data/Original/originalTrainingCoordinatesId',originalTrainingCoordinatesId)
    np.save('./Data/testingApFingerprints',testingApFingerprints)
    np.save('./Data/testingCoordinates',testingCoordinates)
    np.save('./Data/testingCoordinatesId',testingCoordinatesId)
    np.save('./Data/Original/originalTestingApFingerprints',originalTestingApFingerprints)
    np.save('./Data/Original/originalTestingCoordinates',originalTestingCoordinates)
    np.save('./Data/Original/originalTestingCoordinatesId',originalTestingCoordinatesId)

    np.save('./Data/Original/rpCoordinatesList',coordinatesList)
    np.save('./Data/Original/apList',apList)
    print('保存完毕！')
