# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:08:56 2018

@author: Winham

# CPSC_extract_features.py: 对记录提取HRV特征以及年龄性别信息

"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from CPSC_model import ManFeat_HRV
from CPSC_config import Config
import CPSC_utils as utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
warnings.filterwarnings("ignore")
config = Config()

records_name = np.array(os.listdir(config.DATA_PATH))
records_label = np.load(config.REVISED_LABEL) - 1
class_num = len(np.unique(records_label))
age_gen = pd.read_csv(config.AGE_GEN_INFO)
AGE_GEN_DIMENSION = age_gen.shape[0]

train_val_records, test_records, train_val_labels, test_labels = train_test_split(
    records_name, records_label, test_size=0.2, random_state=config.RANDOM_STATE)
train_records, val_records, train_labels, val_labels = train_test_split(
    train_val_records, train_val_labels, test_size=0.2, random_state=config.RANDOM_STATE)

train_records, train_labels = utils.oversample_balance(train_records, train_labels, config.RANDOM_STATE)
val_records, val_labels = utils.oversample_balance(val_records, val_labels, config.RANDOM_STATE)

# 对训练集提取特征 -----------------------------------------------------------------------------------------------------
man_features_r = np.zeros([len(train_records), ManFeat_HRV.FEAT_DIMENSION + AGE_GEN_DIMENSION])
for i in range(len(train_records)):
    print('Process train No.' + str(i+1) + '/' + str(len(train_records)))
    sig = np.load(config.DATA_PATH + train_records[i])[1, :]
    Feat_HRV = ManFeat_HRV(sig, config.Fs)
    feat_hrv = Feat_HRV.extract_features()
    feat_a_g = np.array(age_gen[train_records[i]])
    man_features_rt = np.concatenate((feat_hrv, feat_a_g))
    man_features_r[i] = man_features_rt
    del sig, Feat_HRV, feat_hrv, feat_a_g, man_features_rt

# 对验证集提取特征 -----------------------------------------------------------------------------------------------------
man_features_v = np.zeros([len(val_records), ManFeat_HRV.FEAT_DIMENSION + AGE_GEN_DIMENSION])
for i in range(len(val_records)):
    print('Process val No.' + str(i+1) + '/' + str(len(val_records)))
    sig = np.load(config.DATA_PATH + val_records[i])[1, :]
    Feat_HRV = ManFeat_HRV(sig, config.Fs)
    feat_hrv = Feat_HRV.extract_features()
    feat_a_g = np.array(age_gen[val_records[i]])
    man_features_vt = np.concatenate((feat_hrv, feat_a_g))
    man_features_v[i] = man_features_vt
    del sig, Feat_HRV, feat_hrv, feat_a_g, man_features_vt

# 对测试集提取特征 -----------------------------------------------------------------------------------------------------
man_features_t = np.zeros([len(test_records), ManFeat_HRV.FEAT_DIMENSION + AGE_GEN_DIMENSION])
for i in range(len(test_records)):
    print('Process test No.' + str(i+1) + '/' + str(len(test_records)))
    sig = np.load(config.DATA_PATH + test_records[i])[1, :]
    Feat_HRV = ManFeat_HRV(sig, config.Fs)
    feat_hrv = Feat_HRV.extract_features()
    feat_a_g = np.array(age_gen[test_records[i]])
    man_features_tt = np.concatenate((feat_hrv, feat_a_g))
    man_features_t[i] = man_features_tt
    del sig, Feat_HRV, feat_hrv, feat_a_g, man_features_tt

# 保存特征集 -----------------------------------------------------------------------------------------------------------
np.save(config.MAN_FEATURE_PATH + 'man_features_r.npy', man_features_r)
np.save(config.MAN_FEATURE_PATH + 'man_features_v.npy', man_features_v)
np.save(config.MAN_FEATURE_PATH + 'man_features_t.npy', man_features_t)
