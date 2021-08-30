# make sure to run this file in the same folder as sdegrad.py, electricdata.pkl
import pickle
import numpy as np
from sdegrad import sde
import tensorflow as tf

#%%
with open('electricdata.pkl', 'rb') as f:
    data = pickle.load(f)
# excerpt of data from https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
# ~3.5 years of electrical usage from 20 customers in 15 minute intervals
# data is 2d numpy array, rows is time, columns are features (customers)

# split into train/test
split_ratio = .2
split_ind = int(len(data)*(1-split_ratio))
train = data[:split_ind]
test = data[split_ind:]

# normalize
mean = np.mean(train, axis=0)
std = np.std(train, axis=0)
train = (train-mean)/std
test = (test-mean)/std

# add periodicity: in days and years
time = np.arange(0, len(data))
dayperiod = time/(24*4)*2*np.pi
yearperiod = time/(24*4*365)*2*np.pi
periods = np.concatenate([np.expand_dims(np.sin(dayperiod),1), np.expand_dims(np.cos(dayperiod),1),
           np.expand_dims(np.sin(yearperiod),1), np.expand_dims(np.cos(yearperiod),1)], axis=1)

train = np.concatenate([train, periods[:split_ind]], axis=1)
test = np.concatenate([test, periods[split_ind:]], axis=1)

# convert to tensor
train = tf.convert_to_tensor(train)
test = tf.convert_to_tensor(test)

#%% make model and optimizer

class mysde(sde):
    # Just the sde class, with periodicity added
    @tf.function
    def add_periodic_input_to_curstate(self, curstate, t):
        batch_size = tf.shape(curstate)[0]
        dayperiod = t/(24*4)*2*3.1415926535
        yearperiod = t/(24*4*365)*2*3.1415926535
        temp = tf.tile(tf.expand_dims(tf.stack([tf.math.sin(dayperiod), tf.math.cos(dayperiod),
            tf.math.sin(yearperiod), tf.math.cos(yearperiod)]),0),[batch_size,1])
        return tf.concat([curstate, temp],0)


model = mysde(20, pastlen=12)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#%% training
batch_size = 8
prediction_length = 12
nbatches = int(len(train)/prediction_length)*10

for i in range(nbatches):
    ind = int(np.random.rand()*(len(train)-model.pastlen))+model.pastlen  # index of first prediction
    curdata = [tf.tile(train[j+ind-model.pastlen:j+ind-model.pastlen+1,:], [batch_size, 1]) for j in range(prediction_length+model.pastlen)]

    init_state = curdata[:model.pastlen]
    yhat = curdata[model.pastlen:]

    obj, grad = model.grad(init_state, prediction_length, yhat, start=ind)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))

    if i % 100 == 0:
        print('objective value for batch '+str(i)+' is '+str(obj))






