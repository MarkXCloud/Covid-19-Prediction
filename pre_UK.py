import os
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LeakyReLU, PReLU
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def deltaarr(x):
    delta = []
    for i in range(len(x) - 1):
        delta.append(x[i + 1] / x[i])
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


np.random.seed(1337)
average = 0
standard = 0
Epochs = 2000
data = [
    13, 13, 15, 20, 23,
    36, 36, 51, 85, 116,
    163, 206, 206, 319, 373,
    460, 590, 798, 1140, 1372,
    1543, 1950, 2626, 3269, 3983,
    5018, 5683, 6650, 8077, 9529,
    11772, 14579, 17089, 19522, 22141,
    25150, 29474, 33718, 38168, 41903,
    47806, 51608, 55242, 60733, 65077,
    73758,78991
]
#b = deltaarr(data)
b = np.log(data)
x_train = np.linspace(1, len(b), len(b))
y_train = normalize(b)

model = Sequential()
model.add(Dense(8, input_shape=(1,)))
model.add(Dense(16))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(Dense(8))
model.add(Activation("relu"))
model.add(Dense(1))
ADD = keras.optimizers.Adadelta(lr=1.2, rho=0.93, epsilon=None, decay=0.0)
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='model_UK.h5',
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
                 batch_size=20,
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
model.save('model_UK.h5')
del model
model = load_model('model_UK.h5')
plt.subplot("413")
x_predict = np.linspace(1, len(b) + 7, len(b) + 7)
y_predict = model.predict(x_predict)
plt.plot(x_predict, y_predict, 'r')
plt.plot(x_train, y_train, 'k')
#standard *= y_predict.std() / y_train.std()
y_predict[-8:] = np.exp(y_predict[-8:] * standard + average)

print(y_predict[-8:])
"""for i in range(-7,0):
    print(data[-1]*(y_predict[i]-1))
    data.append(data[-1]*y_predict[i])"""
plt.subplot("414")
loss = hist.history["mean_absolute_error"]
print(len(loss))
plt.plot(np.arange(1, Epochs + 1), loss)
plt.show()
print(len(b))
