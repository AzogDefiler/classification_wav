
# coding: utf-8

# In[1]:

# Aminov Rezo
import numpy as np
import wave
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from glob import glob
import random

import struct

from keras.models import *
from keras.layers import *
from keras.callbacks import *

import librosa
import soundfile as sf
from keras.models import load_model

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split


# In[2]:

DATA_DIR = 'data_v_7_stc'
meta_file = "{}/meta/meta.txt".format(DATA_DIR)
df = pd.read_csv(meta_file, sep='\t',header=None)
labels_name = df[4].unique()


# In[3]:

# кодирование лейблов
onehot_dict = {}
for ii, lab in enumerate(labels_name):
    y_ = np.zeros(len(labels_name))
    y_[ii] = 1
    onehot_dict.update({lab:ii})
    
# обратный хэш
hot_to_one = {}
for k,v in onehot_dict.items():
    hot_to_one.update({v:k})


# In[4]:

# экстрактор фич: Мел-кепстральные коэффициенты (MFCC). https://habr.com/post/140828/
def extract_feature(file_name):
    X, sample_rate = sf.read(file_name, dtype='float32')
    if X.ndim > 1:
        X = X[:,0]
    X = X.T
    # преобразование Фурье
    stft = np.abs(librosa.stft(X))
    # MFCC
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    # мэл спектр
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    # спектр-ный контраст
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


# In[5]:

files_test = glob(DATA_DIR+'/test/*.wav')


# In[6]:

model = load_model('weights/model.hdf5')
model.load_weights('weights/model_weights.hdf5')


# In[7]:

CNT=0 # кол-во всех не 'unknown', подмножество 'A'
GOOD=0 # кол-во правильно опред-ых файлов в подмножестве 'A'
BAD=0 # кол-во не правильно опред-ых файлов в подмножестве 'A'

filew = open("result.txt","a") 
features_test = np.empty((0,193))
for file in files_test:    
    try:
        mfccs, chroma, mel, contrast,tonnetz = extract_feature(file)
    except Exception as e:
        print("[Error] extract feature error. %s" % (e))
        continue
    ext_features_test = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
#     features_test = np.vstack([features_test,ext_features_test])
    pred = model.predict(np.expand_dims([ext_features_test],axis=2))
    score = pred.max()
    class_ = hot_to_one[np.argmax(pred)]
    filename = file.split('/')[2]
    
    filew.write(filename+'\t'+str(score)+'\t'+class_+'\n')
    print(filename+' '+str(score)+' '+class_)
    
    # если файл не 'unknown', делаю подсчет совпадений лейбла и наз. файла
    # примерный подсчет, т.к. неизвестно к какому классу относятся файлы
    # с наз. 'unknown'
    if 'unknown' not in filename:
        CNT+=1
        if class_ in filename:
            GOOD+=1
        else:
            BAD+=1
    
filew.close()


# In[8]:

CNT, GOOD, BAD


# In[9]:

GOOD/CNT


# In[ ]:



