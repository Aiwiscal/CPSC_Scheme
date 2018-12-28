# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 20:26:27 2018

@author: Winham

# CPSC_utils.py: 辅助函数模块

"""

import numpy as np
import pywt


def WTfilt_1d(sig):
    """
    # 使用小波变换对单导联ECG滤波
    # 参考：Martis R J, Acharya U R, Min L C. ECG beat classification using PCA, LDA, ICA and discrete
    wavelet transform[J].Biomedical Signal Processing and Control, 2013, 8(5): 437-448.
    :param sig: 1-D numpy Array，单导联ECG
    :return: 1-D numpy Array，滤波后信号
    """
    coeffs = pywt.wavedec(sig, 'db6', level=9)
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    sig_filt = pywt.waverec(coeffs, 'db6')
    return sig_filt


def SegSig_1d(sig, seg_length=1500, overlap_length=0,
          full_seg=True, stt=0):
    """
    # 按指定参数对单导联ECG进行切片
    :param sig: 1-D numpy Array，单导联ECG
    :param seg_length:  int，切片的采样点长度
    :param overlap_length: int, 切片之间相互覆盖的采样点长度，默认为0
    :param full_seg:  bool， 是否对信号末尾不足seg_length的片段进行延拓并切片，默认True
    :param stt:  int, 开始进行切片的位置， 默认从头开始（0）
    :return: 2-D numpy Array, 切片个数 * 切片长度
    """
    length = len(sig)
    SEGs = np.zeros([1, seg_length])
    start = stt
    while start+seg_length <= length:
        tmp = sig[start:start+seg_length].reshape([1, seg_length])
        SEGs = np.concatenate((SEGs, tmp))
        start += seg_length
        start -= overlap_length
    if full_seg:
        if start < length:
            pad_length = seg_length-(length-start)
            tmp = np.concatenate((sig[start:length].reshape([1, length-start]),
                                sig[:pad_length].reshape([1, pad_length])), axis=1)
            SEGs = np.concatenate((SEGs, tmp))
    SEGs = SEGs[1:]
    return SEGs


def Pad_1d(sig, target_length):
    """
    # 对小于target_length的信号进行补零
    :param sig: 1-D numpy Array，输入信号
    :param target_length: int，目标长度
    :return:  1-D numpy Array，输出补零后的信号
    """
    pad_length = target_length - sig.shape[0]
    if pad_length > 0:
        sig = np.concatenate((sig, np.zeros(int(pad_length))))
    return sig


def Stack_Segs_generate(sig, seg_num=24, seg_length=1500, full_seg=True, stt=0):
    """
    # 对单导联信号滤波，按照指定切片数目和长度进行切片，并堆叠为矩阵
    :param sig: 1-D numpy Array, 输入单导联信号
    :param seg_num: int，指定切片个数
    :param seg_length: int，指定切片采样点长度
    :param full_seg: bool，是否对信号末尾不足seg_length的片段进行延拓并切片，默认True
    :param stt: int, 开始进行切片的位置， 默认从头开始（0）
    :return: 3-D numpy Array, 1 * 切片长度 * 切片个数
    """
    sig = WTfilt_1d(sig)
    if len(sig) < seg_length+seg_num:
        sig = Pad_1d(sig, target_length=(seg_length+seg_num-1))
        
    overlap_length = int(seg_length-(len(sig) - seg_length)/(seg_num-1))
    
    if (len(sig) - seg_length) % (seg_num-1) == 0:
        full_seg = False
 
    SEGs = SegSig_1d(sig, seg_length=seg_length,
                         overlap_length=overlap_length, full_seg=full_seg, stt=stt)
    del sig
    SEGs = SEGs.transpose()
    SEGs = SEGs.reshape([1, SEGs.shape[0], SEGs.shape[1]])
    return SEGs


def Fetch_Pats_Lbs_sLead(Pat_files, Path, target_lead=1, seg_num=24,
                         seg_length=1500, full_seg=True, stt=0, buf_size=100):
    """
    # 对指定病人的单导联信号进行滤波，按照指定切片数目和长度进行切片，并堆叠为矩阵
    :param Pat_files: list or 1-D numpy Array, 指定病人文件
    :param Path: str，数据存放路径
    :param target_lead: int，指定单导联，例如1指II导联
    :param seg_num: int，指定切片个数
    :param seg_length: int，指定切片采样点长度
    :param full_seg: bool，是否对信号末尾不足seg_length的片段进行延拓并切片，默认True
    :param stt: int, 开始进行切片的位置， 默认从头开始（0）
    :param buf_size: 用于加速过程的缓存Array大小，默认为100
    :return:
    """
    seg_length = int(seg_length)
    SEG_buf = np.zeros([1, seg_length, seg_num])
    SEGs = np.zeros([1, seg_length, seg_num])
    for i in range(len(Pat_files)):
        sig = np.load(Path+Pat_files[i])[target_lead, :]
        SEGt = Stack_Segs_generate(sig, seg_num=seg_num,
                 seg_length=seg_length, full_seg=full_seg, stt=stt)
        SEG_buf = np.concatenate((SEG_buf, SEGt))
        del SEGt
        if SEG_buf.shape[0] >= buf_size:
            SEGs = np.concatenate((SEGs, SEG_buf[1:]))
            del SEG_buf
            SEG_buf = np.zeros([1, seg_length, seg_num])
    if SEG_buf.shape[0] > 1:
        SEGs = np.concatenate((SEGs, SEG_buf[1:]))
    del SEG_buf
    return SEGs[1:]


def oversample_balance(records, labels, rand_seed):
    """
    # 通过随机过采样使各类样本数目平衡
    :param records: 1-D numpy Array，不平衡样本记录名集合
    :param labels: 1-D numpy Array，对应标签
    :param rand_seed：int, 随机数种子
    :return: 平衡后的记录名集合和对应标签
    """
    class_num = len(np.unique(labels))
    num_records = len(records)
    num_categories = []
    for i in range(class_num):
        num_categories.append(len(labels[labels == i]))
    upsample_rate = max(num_categories)/np.array(num_categories)-1
    for i in range(class_num):
        rate = upsample_rate[i]
        if rate < 1 and rate > 0:
            records_this_class = records[labels == i]
            oversample_size = int(np.ceil(num_categories[i]*rate))
            np.random.seed(rand_seed)
            rand_sample = np.random.choice(records_this_class, 
                                           size=oversample_size,
                                           replace=False)
            records = np.concatenate((records, rand_sample))
            labels = np.concatenate((labels, np.ones(oversample_size)*i))
    over_sample_records = []
    over_sample_labels = []
    for i in range(num_records):
        rate = upsample_rate[int(labels[i])]
        if rate >= 1:
            over_sample_records = over_sample_records + [records[i]] * int(round(rate))
            over_sample_labels = over_sample_labels + [labels[i]] * int(round(rate))
            
    records = np.concatenate((records, np.array(over_sample_records)))
    labels = np.concatenate((labels, np.array(over_sample_labels)))
    return records, labels
