from common import load_track, GENRES
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Dropout, Activation, \
        TimeDistributed, Convolution1D, MaxPooling1D, BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from optparse import OptionParser
from sys import stderr, argv
import os
import keras
from keras.models import load_model
from train_model import BATCH_SIZE

from create_data_pickle import get_default_shape

import easygui
testset_path=easygui.fileopenbox()




savedModel=os.path.join(os.getcwd(),'models/model.h5')
model = load_model(savedModel)


savedData=os.path.join(os.getcwd(),'data/data.pkl')
with open(savedData, 'rb') as f:
    data = pickle.load(f)


labels = {0:"blues",1:"classical",2:'country', 3:'disco',4:'hiphop',5:'jazz',6:'metal',
        7:'pop',8:'reggae',9:'rock'}
# to feed model

#generator= train_datagen.flow_from_directory("data/genres", batch_size=BATCH_SIZE)
#label_map = (generator.class_indices)

#lb = pickle.loads(open(args["labelbin"], "rb").read())

# classify the input audio



x = data['x']

y = data['y']
t=data['track_paths']
print("classifying audio...")

'''
#############test code#########


print(x.shape)
print("")
print(x[0].shape)
print("")
print("h1")
print(x)
print("h2")
print("")
print(x[0])
proba = model.predict(x)[0]
proba2 = model.predict(x)

print(proba)
#y_classes=keras.utils.np_utils.probas_to_classes(proba)
#print(y_classes)


idx = np.argmax(proba)
aud=np.argmax(x[0])
#label = t.classes_[idx]
print(aud,"----->",t[idx])


i=121
predicted_classes = np.argmax(np.round(proba2),axis=1)
index=predicted_classes[i]
print("Prediction for ",t[i]," is ",labels[index])

#############test code ends#########
'''


dataset_path=os.getcwd()+"/data/genres"

default_shape=get_default_shape(dataset_path)
print(default_shape)

TRACK_COUNT = 1000
test_pos=TRACK_COUNT+100
t1=np.zeros((TRACK_COUNT+200,) + default_shape, dtype=np.float32)
t1=x
#t1=t1.resize(2000,647,128)
#testset_path=os.getcwd()+'/test'
print(testset_path)
file_name='blues.00000.au'
print('Processing', file_name)
#path = os.path.join(testset_path, file_name)

t1, _ = load_track(testset_path, default_shape)




pred1=model.predict(np.array([t1]))[0]

predict_class=np.argmax(np.round(pred1))
index=predict_class

print("Prediction for the selected song is ",labels[index])
