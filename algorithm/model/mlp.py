import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop,Adam

class Discriminator(object):
    def __init__(self,xs,num_classes):
        self.xs=xs
        self.ngpu=2
        self.num_classes=num_classes

    def get_model(self):
        model = Sequential()
        model.add(Dense(256, activation='tanh', input_shape=(44 * 44,)))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        return model