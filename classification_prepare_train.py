
# coding: utf-8

# In[10]:

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


# In[3]:

DATA_DIR = 'data_v_7_stc'


# In[4]:

test_file = "{}/audio/background_0007.wav".format(DATA_DIR)

with wave.open(test_file, 'rb') as f:
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    print(nchannels, sampwidth, framerate, nframes) 
    strData = f.readframes(nframes)

waveData = np.fromstring(strData, dtype=np.int16)

print(waveData[:20])

waveData_norm = waveData * 1.0 / (max(abs(waveData)))

time = np.arange(0, nframes)*(1.0 / framerate)
plt.plot(time, waveData_norm)
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title("Single channel wavedata")
plt.grid('on')
plt.show()


# ## Load DF

# In[5]:

meta_file = "{}/meta/meta.txt".format(DATA_DIR)


# In[6]:

df = pd.read_csv(meta_file, sep='\t',header=None)


# In[7]:

df.head()


# In[9]:

# mean time
df[3].mean()


# In[8]:

# все уникальные лейблы
labels_name = df[4].unique()


# In[10]:

# кодирование лейблов
onehot_dict = {}
for ii, lab in enumerate(labels_name):
    y_ = np.zeros(len(labels_name))
    y_[ii] = 1
    onehot_dict.update({lab:ii})


# In[15]:

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


# In[11]:

features, labels = np.empty((0,193)), np.empty(0)
for file, label in zip(df[0],df[4]):    
    try:
        mfccs, chroma, mel, contrast,tonnetz = extract_feature(DATA_DIR+'/audio/'+file)
    except Exception as e:
        print("[Error] extract feature error. %s" % (e))
        continue
    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    features = np.vstack([features,ext_features])
    labels = np.append(labels, onehot_dict[label])
    print("extract %s features done" % (file))


# In[12]:

features = np.array(features)
labels = np.array(labels)


# In[13]:

np.save('feat.npy', features)
np.save('label.npy', labels)


# In[12]:

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split


# In[40]:

# Prepare the data
X = np.load('feat.npy')
y = np.load('label.npy').ravel()

# X = features
# y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=233)


# In[17]:

# 3 конв. слоя, 2 полносвяз. слоя

model = Sequential()

model.add(Conv1D(64, 3, activation='relu', input_shape=(193, 1)))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(256, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))


# model.add(Conv1D(64, 3, activation='relu', input_shape=(193, 1)))
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


# In[19]:

model.summary()


# In[42]:

y_train = keras.utils.to_categorical(y_train , num_classes=8)
y_test = keras.utils.to_categorical(y_test , num_classes=8)

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)


# In[21]:

res = model.fit(X_train, y_train, batch_size=32,validation_data=(X_test, y_test), epochs=50)


# In[22]:

# accuracy
plt.plot(res.history['acc'])
plt.plot(res.history['val_acc'])


# In[23]:

# loss
plt.plot(res.history['loss'])
plt.plot(res.history['val_loss'])


# In[24]:

# оценка модели на тестовых данных
score, acc = model.evaluate(X_test, y_test, batch_size=16)


# In[25]:

score, acc


# In[38]:

model.save('weights/model.hdf5')
model.save_weights('weights/model_weights.hdf5')


# In[ ]:



