import os
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LeakyReLU, PReLU

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def deltaarr(x):
    delta = []
    for i in range(len(x) - 1):
        delta.append(x[i + 1] - x[i])
    return np.array(delta)


def normalize(x):
    global average, standard
    mean = x.mean()
    average = mean
    std = x.std()
    standard = std
    normal = (x - mean) / std
    return normal


def random_sample(x, y):
    sr = 25
    sample = []
    test = []
    r = np.random.randint(1, 100, len(x))
    for i in range(len(x)):
        if r[i] <= sr:
            sample.append(x[i])
            test.append(y[i])
    sample = np.array(sample)
    test = np.array(test)
    return sample, test


def smooth(x):
    sample = []
    sample.append(x[0])
    for i in range(1, len(x)):
        sample.append(0.1 * x[i - 1] + 0.9 * x[i])
    return np.array(sample)


import numpy as np

np.random.seed(1337)
average = 0
standard = 0
Epochs = 1300
data = [20, 79, 157, 230,
     323, 470, 653, 888, 1128,
     1694, 2036, 2502, 3089, 3858,
     4636, 5883, 7375, 9172, 10149,
     12462, 15113, 17660, 21157, 24747,
     27980, 31506, 35713, 41035, 47021,
     53578, 59138, 64378, 69176, 74386,
     81129, 86498, 93051, 97689, 101739,
     105792, 110574, 115242, 119827, 125016,
     128948, 132547, 135586, 139422,147577,
     152860
     ]
b = smooth(deltaarr(data))
x_train = np.linspace(1, len(b), len(b))
y_train = normalize(b)

model = Sequential()
model.add(Dense(8, input_shape=(1,)))
model.add(LeakyReLU(alpha=0.05))
model.add(Dense(16))
model.add(LeakyReLU(alpha=0.05))
model.add(Dense(16,activation="relu"))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(Dense(8,activation="relu"))
model.add(Activation("relu"))

model.add(Dense(1))
ADD = keras.optimizers.Adadelta(lr=1.2, rho=0.93, epsilon=None, decay=0.0)
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='model_Italy.h5',
        monitor='val_mse',
        save_best_only=True,
    ),
    keras.callbacks.TensorBoard(
        log_dir='my_log',
        histogram_freq=1,
    )
]
model.compile(optimizer=ADD, loss='mse', metrics=['mae'])
model.summary()
hist = model.fit(x_train,
                 y_train,
                 epochs=Epochs,
                 batch_size=15,
                 verbose=0,
                 validation_split=0.15,
                 callbacks=callbacks_list
                 )

print(hist.history)

plt.subplot("411")
plt.plot(x_train, y_train, 'k')

plt.subplot("412")
x_predict = np.linspace(1, len(b) + 7, len(b) + 7)
y_predict = model.predict(x_predict)
plt.plot(x_predict, y_predict, 'r')
model.save('model_Italy.h5')
del model
model = load_model('model_Italy.h5')

plt.subplot("413")
x_predict = np.linspace(1, len(b) + 7, len(b) + 7)
y_predict = model.predict(x_predict)
plt.plot(x_predict, y_predict, 'r')
plt.plot(x_train, y_train, 'k')

standard *= y_predict.std() / y_train.std()
average *= 1 + y_predict.mean()
print(y_predict[-7:] * standard + average)
plt.subplot("414")
loss = hist.history["mean_absolute_error"]
print(len(loss))
plt.plot(np.arange(1, Epochs + 1), loss)
plt.show()
print(len(b))
