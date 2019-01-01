# CPSC_Scheme

## Blog:
https://blog.csdn.net/qq_15746879
一个不成熟的开源小方案：关于2018中国生理信号挑战赛（CPSC-2018）(1)-(4)

## Environment：
biosppy==0.6.1
h5py==2.6.0
keras==2.2.4
numpy==1.15.4
pandas==0.19.2
pyentrp==0.5.0
PyWavelets==1.0.1
scikit-learn==0.18.1
tensorflow-gpu==1.9.0
xgboost==0.81

## warning:
A CUDA device is required if you want to use the models shared in this repo.
If you don't have a CUDA device, change the "CuDNNLSTM" in CPSC_model.py to "LSTM" and train it using CPUs.But it may be very slow.
