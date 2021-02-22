import os
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def smooth(x):
    sample = []
    sample.append(x[0])
    for i in range(1, len(x)):
        sample.append(0.1 * x[i - 1] + 0.9 * x[i])
    return np.array(sample)


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
    r = np.random.randint(1, 100, size=100)
    sample = []
    test = []
    for i in range(len(x)):
        if r[i] <= sr:
            sample.append(x[i])
            test.append(y[i])
    sample = np.array(sample)
    test = np.array(test)
    return sample, test


np.random.seed(1337)

data = [20, 79, 157, 230,
     323, 470, 653, 888, 1128,
     1694, 2036, 2502, 3089, 3858,
     4636, 5883, 7375, 9172, 10149,
     12462, 15113, 17660, 21157, 24747,
     27980, 31506, 35713, 41035, 47021,
     53578, 59138, 64378, 69176, 74386,
     81129, 86498, 93051, 97689, 101739,
     105792, 110574, 115242, 119827, 125016,
     128948, 132547, 135586, 139422, 147577,
     152271
     ]
b = smooth(deltaarr(data))
x_train = np.linspace(1, len(b), len(b))
y_train = normalize(b)

model = load_model('model_Italy.h5')
plt.subplot("411")
plt.plot(x_train, y_train, 'k')

plt.subplot("412")
x_predict = np.linspace(1, len(b) + 7, len(b) + 7)
y_predict = model.predict(x_predict)
plt.plot(x_predict, y_predict, 'r')

plt.subplot("413")
x_predict = np.linspace(1, len(b) + 7, len(b) + 7)
y_predict = model.predict(x_predict)
plt.plot(x_predict, y_predict, 'r')
plt.plot(x_train, y_train, 'k')

standard *= y_predict.std() / y_train.std()
average *= 1 + y_predict.mean()
print(y_predict[-8:] * standard + average)
plt.show()
print(len(b))
