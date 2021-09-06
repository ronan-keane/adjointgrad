# make sure to run this file in the same folder as sdegrad.py, electricdata.pkl
import pickle
import numpy as np
from sdegrad import sde, sde_mle, jump_ode
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
train = tf.convert_to_tensor(train, dtype=tf.float32)
test = tf.convert_to_tensor(test, dtype=tf.float32)

#%% make model

# class mysde(sde):
# class mysde(sde_mle):
class mysde(jump_ode):
    # Just the sde class, with periodicity added
    @tf.function
    def add_periodic_input_to_curstate(self, curstate, t):
        batch_size = tf.shape(curstate)[0]
        dayperiod = t/(24*4)*2*3.1415926535
        yearperiod = t/(24*4*365)*2*3.1415926535
        temp = tf.tile(tf.expand_dims(tf.stack([tf.math.sin(dayperiod), tf.math.cos(dayperiod),
            tf.math.sin(yearperiod), tf.math.cos(yearperiod)]),0),[batch_size,1])
        return tf.concat([curstate, temp],1)


model = mysde(20, 3, pastlen=12, p=1e-4)

#%% training
# used for huber loss
# batch_size_list = [8, 8, 8, 8, 8]
# learning_rate_list = [1e-4, 1e-4, 5e-5, 1e-5, 1e-6]
# prediction_length_list = [12, 24, 48, 96, 192]  # 3 hours up to 2 days
# nbatches_list = [2000, 2000, 2000, 2000, 2000]
# for mle loss
# batch_size_list = [8, 8, 8, 8, 8, 8, 8]
# learning_rate_list = [1e-4, 1e-3, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
# prediction_length_list = [12, 12, 12, 24, 48, 96, 192]  # 3 hours up to 2 days
# nbatches_list = [2000, 2000, 2000, 2000, 2000, 2000]
# for jump_ode
batch_size_list = [8, 8, 8, 8, 8]
learning_rate_list = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
prediction_length_list = [12, 24, 48, 96, 192]  # 3 hours up to 2 days
nbatches_list = [2000, 2000, 2000, 2000, 2000]

assert len(batch_size_list) == len(learning_rate_list) == len(prediction_length_list) == len(nbatches_list)

for j in range(len(batch_size_list)):
    batch_size = batch_size_list[j]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_list[j])
    prediction_length = prediction_length_list[j]
    nbatches = nbatches_list[j]

    for i in range(nbatches):
        ind = int(np.random.rand()*(len(train)-model.pastlen-prediction_length))+model.pastlen  # index of first prediction
        curdata = [tf.tile(train[j+ind-model.pastlen:j+ind-model.pastlen+1,:], [batch_size, 1]) for j in range(prediction_length+model.pastlen)]

        init_state = curdata[:model.pastlen]
        yhat = curdata[model.pastlen:]

        obj, grad = model.grad(init_state, prediction_length, yhat, start=ind)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        if i % 100 == 0:
            print('objective value for batch '+str(i)+' is '+str(obj))


#%% make baseline model which uses average electric in that time at that day of the week
period = 384
baseline_predictions = [np.mean(train[i::period], axis=0) for i in range(period)]
def baseline(ind):
    ind = (ind+len(train))%(period)
    return baseline_predictions[ind]

#%% test
import matplotlib.pyplot as plt
batch_size = 1  # number of replications
prediction_length = 24*4*7
ind = 12  # starting time in test set
customer = 10  # which customer to plot

offset = len(train)
assert ind-model.pastlen >=0
assert ind + prediction_length <= len(test)
assert customer < 20

curdata = [tf.tile(test[j+ind-model.pastlen:j+ind-model.pastlen+1,:], [batch_size, 1]) for j in range(prediction_length+model.pastlen)]

init_state = curdata[:model.pastlen]
yhat = curdata[model.pastlen:]

obj = model.solve(init_state, prediction_length, yhat, start=ind+offset)

y1 = [yhat[i][0,customer] for i in range(len(yhat))]
x = [[model.mem[i+model.pastlen][j,customer] for i in range(len(yhat))] for j in range(batch_size)]
base_y = [baseline(i)[customer] for i in range(ind, ind+prediction_length)]

plt.plot(y1)
print('baseline mse is '+str(np.mean(np.square(np.array(y1)-np.array(base_y)))))
print('sde mse is '+str(np.mean(np.square(np.array(x[0])-np.array(y1)))))
plt.plot(base_y)
for i in range(1):
    plt.plot(x[i])

#%% fancy plot
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

x = np.zeros((prediction_length, batch_size))
for i in range(len(x)):
    x[i,:] = model.mem[i+model.pastlen][:,customer]

mean = np.mean(x,axis=1)
std = np.std(x,axis=1)
t = list(range(ind+len(train), ind+len(train)+prediction_length))

plt.figure(figsize=(10,8))
frame = plt.gca()
plt.plot(t, y1, 'C0')
plt.fill_between(t, mean-std, mean+std, alpha=.3, facecolor='C2')
plt.plot(t, base_y, 'C1')
plt.plot(t, x[:,0], 'C2', alpha=.3)
frame.axes.get_xaxis().set_visible(False)
plt.ylabel('normalized demand')
plt.xlabel('time (7 days total)')

legend_elements = [Line2D([0], [0], color='C0', label='ground truth'),
                   Patch(facecolor='C2', alpha=.3, label='Huber SDE mean '+u'\u00b1'+' std dev'),
                   Line2D([0], [0], color = 'C2', alpha=.3, label = 'Huber SDE example prediction'),
                   Line2D([0], [0], color='C1', label = 'historical average')]

frame.legend(handles=legend_elements)


