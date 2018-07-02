
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


# In[2]:

DATA_DIR = 'data_v_7_stc'


# In[3]:

def loadwav(file):
    test_file = DATA_DIR+"/audio/"+file+".wav"

    with wave.open(test_file, 'rb') as f:
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        print(nchannels, sampwidth, framerate, nframes) 
        strData = f.readframes(nframes)

    waveData = np.fromstring(strData, dtype=np.int16)

    print(waveData[:20])

    waveData_norm = waveData * 1.0 / (max(abs(waveData)))

    time = np.arange(0, nframes)*(1.0 / framerate)
    return time, waveData_norm
    
time, waveData_norm = loadwav('bg_0048_time_stretch_0')
plt.plot(time, waveData_norm)
plt.grid('on')
plt.show()

time1, waveData_norm1 = loadwav('knocking_door_0037_time_stretch_5')
plt.plot(time1, waveData_norm1)
plt.grid('on')
plt.show()

time1, waveData_norm1 = loadwav('speech_0038')
plt.plot(time1, waveData_norm1)
plt.grid('on')
plt.show()


# ## Load DF

# In[4]:

meta_file = "{}/meta/meta.txt".format(DATA_DIR)


# In[5]:

df = pd.read_csv(meta_file, sep='\t',header=None)


# In[6]:

len(df)


# In[7]:

df.head()


# In[8]:

# mean time
df[3].mean()


# In[9]:

# все уникальные лейблы
labels_name = df[4].unique()


# In[10]:

for lbl in labels_name:    
    print(lbl, df[df[4] == lbl][3].mean())


# In[11]:

# кодирование лейблов
onehot_dict = {}
for ii, lab in enumerate(labels_name):
    y_ = np.zeros(len(labels_name))
    y_[ii] = 1
    onehot_dict.update({lab:ii})


# In[13]:

# экстрактор фич: Мел-кепстральные коэффициенты (MFCC). https://habr.com/post/140828/
def extract_feature(file_name):
    X, sample_rate = sf.read(file_name, dtype='float32')
    if X.ndim > 1:
        X = X[:,0]
    X = X.T
    # преобразование Фурье
    stft = np.abs(librosa.stft(X))
    # MFCC
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128).T,axis=0)
    # chroma
#     chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    # мэл спектр
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    # спектр-ный контраст
#     contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

#     tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,mel


# In[14]:

cnt=0
features, labels = np.empty((0,256)), np.empty(0)
for file, label in zip(df[0],df[4]):
    try:
        cnt+=1
        mfccs, mel = extract_feature(DATA_DIR+'/audio/'+file)
    except Exception as e:
        print("[Error] extract feature error. %s" % (e))
        continue
    ext_features = np.hstack([mfccs,mel])
    features = np.vstack([features,ext_features])
    labels = np.append(labels, onehot_dict[label])
    print(cnt)


# In[160]:

features = np.array(features)
labels = np.array(labels)


# In[13]:

np.save('feat.npy', features)
np.save('label.npy', labels)


# In[162]:

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split


# In[163]:

# Prepare the data
# X = np.load('feat.npy')
# y = np.load('label.npy').ravel()

X = features
y = labels

# X = np.exp(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=233)


# In[164]:

# 3 конв. слоя, 2 полносвяз. слоя

model = Sequential()

model.add(Conv1D(128, 3, activation='relu', input_shape=(256, 1)))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 7, activation='relu'))
model.add(Conv1D(128, 9, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))


# model.add(Conv1D(64, 3, activation='relu', input_shape=(256, 1)))
# model.add(Conv1D(64, 3, activation='relu'))
# model.add(MaxPooling1D(3))
# model.add(Conv1D(128, 3, activation='relu'))
# model.add(Conv1D(128, 3, activation='relu'))
# model.add(GlobalAveragePooling1D())
# model.add(Dropout(0.5))
# model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])#adam


# In[165]:

model.summary()


# In[166]:

y_train = keras.utils.to_categorical(y_train , num_classes=8)
y_test = keras.utils.to_categorical(y_test , num_classes=8)

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)


# In[169]:

res = model.fit(X_train, y_train, batch_size=64,validation_data=(X_test, y_test), epochs=5)


# In[170]:

# accuracy
plt.plot(res.history['acc'])
plt.plot(res.history['val_acc'])


# In[171]:

# loss
plt.plot(res.history['loss'])
plt.plot(res.history['val_loss'])


# In[172]:

# оценка модели на тестовых данных
score, acc = model.evaluate(X_test, y_test, batch_size=16)


# In[173]:

score, acc


# In[179]:

model.save('weights/model.hdf5')
model.save_weights('weights/model_weights.hdf5')


# ## -------------------------------------------------------------------------------------------

# In[174]:

files_test = glob(DATA_DIR+'/test/*.wav')

# обратный хэш
hot_to_one = {}
for k,v in onehot_dict.items():
    hot_to_one.update({v:k})


# In[175]:

CNT=0 # кол-во всех не 'unknown', подмножество 'A'
GOOD=0 # кол-во правильно опред-ых файлов в подмножестве 'A'
BAD=0 # кол-во не правильно опред-ых файлов в подмножестве 'A'

filew = open("result.txt","a") 
features_test = np.empty((0,256))
for file in files_test:    
    try:
        mfccs,mel = extract_feature(file)
    except Exception as e:
        print("[Error] extract feature error. %s" % (e))
        continue
    ext_features_test = np.hstack([mfccs,mel])
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


# In[176]:

CNT, GOOD, BAD


# In[177]:

GOOD/CNT


# In[ ]:




# In[ ]:



