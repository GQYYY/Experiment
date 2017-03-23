#!/usr/bin/env python
# encoding: utf-8

import numpy as np

#AP强度信息的统计类
class apRssStatistics:
	def __init__(self,apFingerprints,rpCoordinates,apList,rpList):
		#输入参数
		self._apFingerprints = apFingerprints  #AP指纹
		self._rpCoordinates = rpCoordinates    #与上面AP指纹对应的RP的坐标
		self._apList = apList                  #整个数据集中AP，有顺序
		self._rpList = rpList                  #整个数据集中的RP，有顺序

		self._Gamma = None
		self._gamma = None
		self._ri = self.__get_Ri()        # Reliability Indicator
		self._delta = self.__get_delta()  # Variance of readings for all RPs

		

	#每个RP的每个AP的信号强度的中位数
	#RP数*AP数
	def __get_gamma(self):
		gamma = np.empty((len(self._rpList),len(self._apList)))
		#判断该RP是否已经统计过，如果已统计，相应位置置为1
		isRpchecked = np.zeros((len(self._rpList),))
		for rpCoordinate in self._rpCoordinates:
			#找到该坐标在rpList中的索引，提供_gamma中的行号
			i = self._rpList.index(rpCoordinate)
			if isRpchecked[i] == 0:
				#每个RP处得到的指纹，100组
				rpFingeprints = self._apFingerprints[np.where(self._rpCoordinates == rpCoordinate)]
				#求每列的中位数，含义是求每个AP对应的信号强度中位值
				gamma[i,:] = np.median(rpFingeprints,axis=0)
				isRpchecked[i] == 1
		return gamma

	#每个RP的每个AP的RSS在所有采集过程中(100次)大于阈值(信号强度中位值)的次数
	#RP数*AP数
	def __get_Gamma(self):		
		Gamma = np.zeros((len(self._rpList),len(self._apList)))
		self._gamma = self.__get_gamma()
		for apFingerprint,rpCoordinate in zip(self._apFingerprints,self._rpCoordinates):
			#找到该apFinerprint所对应的rpCoordinate在rpList中的索引
			i = self._rpList.index(rpCoordinate)
			for j,apRss in enumerate(apFingerprint):
				if apRss >= self._gamma[i,j]:
					Gamma[i,j] += 1
		return Gamma

	#获取每个RP处的每个AP的 Reliability Indicator
	#每一行是一个RP对应的APReliability Indicator向量，用来计算Hamming Distance
	#RP数*AP数
	def __get_ri(self):
		self._Gamma = self.__get_Gamma()
		return np.where(self._Gamma >= 90,1,0)

	#外部接口，获取Ri，Reliability Indicator
	def get_ri(self):
		return self._ri

	#计算每个RP的Rss readings的方差
	#先求每个RP上每个AP的Rss readings的方差，再对所有AP方差求均值
	def __get_delta(self):
		delta = np.zeros((len(self._rpList),len(self._apList)))
		#判断该RP是否已经统计过，如果已统计，相应位置置为1
		isRpchecked = np.zeros((len(self._rpList),))
		for rpCoordinate in self._rpCoordinates:
			#找到该坐标在rpList中的索引，提供_gamma中的行号
			i = self._rpList.index(rpCoordinate)
			if isRpchecked[i] == 0:
				#每个RP处得到的指纹，100组
				rpFingeprints = self._apFingerprints[np.where(self._rpCoordinates == rpCoordinate)]
				#求每列的方差，含义是求每个AP对应的信号强度的方差
				delta[i,:] = np.var(rpFingeprints,axis=0)
				isRpchecked[i] == 1
		return np.mean(delta,axis=1)

	def get_delta(self):
		return self._delta






