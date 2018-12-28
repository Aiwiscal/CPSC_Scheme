# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 13:40:31 2018

@author: Winham

# CPSC_model.py:深度学习网络模型和人工HRV特征提取

"""

import warnings
import numpy as np
from keras.layers import Conv1D, BatchNormalization, Activation, AveragePooling1D, Dense
from keras.layers import Dropout, Concatenate, Flatten, Lambda
from keras import regularizers
from keras.layers import Reshape, CuDNNLSTM, Bidirectional
from biosppy.signals import ecg
from pyentrp import entropy as ent
import CPSC_utils as utils

warnings.filterwarnings("ignore")


class Net(object):
    """
        结合CNN和RNN（双向LSTM）的深度学习网络模型
    """
    def __init__(self):
        pass

    @staticmethod
    def __slice(x, index):
        return x[:, :, index]

    @staticmethod
    def __backbone(inp, C=0.001, initial='he_normal'):
        """
        # 用于信号片段特征学习的卷积层组合
        :param inp:  keras tensor, 单个信号切片输入
        :param C:   double, 正则化系数， 默认0.001
        :param initial:  str, 初始化方式， 默认he_normal
        :return: keras tensor, 单个信号切片经过卷积层后的输出
        """
        net = Conv1D(4, 31, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(inp)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(5, 5)(net)

        net = Conv1D(8, 11, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(5, 5)(net)

        net = Conv1D(8, 7, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(5, 5)(net)

        net = Conv1D(16, 5, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(int(net.shape[1]), int(net.shape[1]))(net)

        return net

    @staticmethod
    def nnet(inputs, keep_prob, num_classes):
        """
        # 适用于单导联的深度网络模型
        :param inputs: keras tensor, 切片并堆叠后的单导联信号.
        :param keep_prob: float, dropout-随机片段屏蔽概率.
        :param num_classes: int, 目标类别数.
        :return: keras tensor， 各类概率及全连接层前自动提取的特征.
        """
        branches = []
        for i in range(int(inputs.shape[-1])):
            ld = Lambda(Net.__slice, output_shape=(int(inputs.shape[1]), 1), arguments={'index': i})(inputs)
            ld = Reshape((int(inputs.shape[1]), 1))(ld)
            bch = Net.__backbone(ld)
            branches.append(bch)
        features = Concatenate(axis=1)(branches)
        features = Dropout(keep_prob, [1, int(inputs.shape[-1]), 1])(features)
        features = Bidirectional(CuDNNLSTM(1, return_sequences=True), merge_mode='concat')(features)
        features = Flatten()(features)
        net = Dense(units=num_classes, activation='softmax')(features)
        return net, features


class ManFeat_HRV(object):
    """
        针对一条记录的HRV特征提取， 以II导联为基准
    """
    FEAT_DIMENSION = 9

    def __init__(self, sig, fs=250.0):
        assert len(sig.shape) == 1, 'The signal must be 1-dimension.'
        assert sig.shape[0] >= fs * 6, 'The signal must >= 6 seconds.'
        self.sig = utils.WTfilt_1d(sig)
        self.fs = fs
        self.rpeaks, = ecg.hamilton_segmenter(signal=self.sig, sampling_rate=self.fs)
        self.rpeaks, = ecg.correct_rpeaks(signal=self.sig, rpeaks=self.rpeaks,
                                         sampling_rate=self.fs)
        self.RR_intervals = np.diff(self.rpeaks)
        self.dRR = np.diff(self.RR_intervals)

    def __get_sdnn(self):  # 计算RR间期标准差
        return np.array([np.std(self.RR_intervals)])

    def __get_maxRR(self):  # 计算最大RR间期
        return np.array([np.max(self.RR_intervals)])

    def __get_minRR(self):  # 计算最小RR间期
        return np.array([np.min(self.RR_intervals)])

    def __get_meanRR(self):  # 计算平均RR间期
        return np.array([np.mean(self.RR_intervals)])

    def __get_Rdensity(self):  # 计算R波密度
        return np.array([(self.RR_intervals.shape[0] + 1) 
                         / self.sig.shape[0] * self.fs])

    def __get_pNN50(self):  # 计算pNN50
        return np.array([self.dRR[self.dRR >= self.fs*0.05].shape[0] 
                         / self.RR_intervals.shape[0]])

    def __get_RMSSD(self):  # 计算RMSSD
        return np.array([np.sqrt(np.mean(self.dRR*self.dRR))])
    
    def __get_SampEn(self):  # 计算RR间期采样熵
        sampEn = ent.sample_entropy(self.RR_intervals, 
                                  2, 0.2 * np.std(self.RR_intervals))
        for i in range(len(sampEn)):
            if np.isnan(sampEn[i]):
                sampEn[i] = -2
            if np.isinf(sampEn[i]):
                sampEn[i] = -1
        return sampEn

    def extract_features(self):  # 提取HRV所有特征
        features = np.concatenate((self.__get_sdnn(),
                self.__get_maxRR(),
                self.__get_minRR(),
                self.__get_meanRR(),
                self.__get_Rdensity(),
                self.__get_pNN50(),
                self.__get_RMSSD(),
                self.__get_SampEn(),
                ))
        assert features.shape[0] == ManFeat_HRV.FEAT_DIMENSION
        return features

