import os
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LeakyReLU, PReLU

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def deltaarr(x):
    delta = []
    for i in range(len(x) - 1):
        delta.append(x[i + 1]-x[i])
    return np.array(delta)


def normalize(x):
    global average, standard
    mean = x.mean()
    average = mean
    std = x.std()
    standard = std
    normal = (x - mean) / std
    return normal

np.random.seed(1337)
average = 0
standard = 0
Epochs = 5000
data = [897, 1408, 2076,
     2857, 4630, 6095, 8149, 9811,
     11901, 14490, 17341, 20530, 24434,
     28138, 31264, 34673, 37289, 40262,
     42747, 44765, 59907, 63950, 66581,
     68595, 70644, 72532, 74284, 74680,
     75571, 76396, 77048, 77269, 77785,
     78195, 78631, 78962, 79972, 80175,
     80303, 80424, 80581, 80734, 80815,
     80868, 80905, 80932, 80969, 80981,
     81007, 81029, 81062, 81099, 81135,
     81202, 81263, 81385, 81457, 81566,
     81691, 81806, 81896, 82034, 82164,
     82282, 82421, 82505, 82601, 82631,
     82691, 82772, 82857, 82899, 82966,
     83039, 83095, 83189, 83246, 83324,
     83400,83524#4.12 latest
     ]
b = np.array(deltaarr(data))
x_train = np.linspace(1, len(b), len(b))
y_train = normalize(b)


model = Sequential()
model.add(Dense(3, input_shape=(1,),activation="relu"))
model.add(Dense(3,activation="relu"))
model.add(Dense(1))
ADD = keras.optimizers.Adadelta(lr=1.2, rho=0.93, epsilon=None, decay=0.0)
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='model_China.h5',
        monitor='val_mse',
        save_best_only=True,
    ),
    keras.callbacks.TensorBoard(
        log_dir='my_log',
        histogram_freq=1,
    )
]
model.summary()
model.compile(optimizer=ADD, loss='mse', metrics=['mae'])

model.summary()
hist = model.fit(x_train,
                 y_train,
                 epochs=Epochs,
                 batch_size=25,
                 verbose=0,
                 validation_split=0.15,
                 callbacks=callbacks_list)

print(hist.history)

plt.subplot("411")
plt.plot(x_train, y_train, 'k')

plt.subplot("412")
x_predict = np.linspace(1, len(b) + 7, len(b) + 7)
y_predict = model.predict(x_predict)
plt.plot(x_predict, y_predict, 'r')
"""model.save('model_China.h5')
del model
model = load_model('model_China.h5')"""
plt.subplot("413")
x_predict = np.linspace(1, len(b) + 7, len(b) + 7)
y_predict = model.predict(x_predict)
plt.plot(x_predict, y_predict, 'r')
plt.plot(x_train, y_train, 'k')
y_predict = y_predict * standard/(len(b)**0.5) + average
print(y_predict[-8:])
"""for i in range(-8,0):
    print(y_predict[i])"""
plt.subplot("414")
loss = hist.history["mean_absolute_error"]
print(len(loss))
plt.plot(np.arange(1, Epochs + 1), loss)
plt.show()
