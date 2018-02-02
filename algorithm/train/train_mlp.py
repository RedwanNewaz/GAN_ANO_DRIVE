from keras.optimizers import RMSprop, Adam
import os
import keras
import pandas as pd
from algorithm.train.base_train import base_train


class train(base_train):
    def __init__(self, batch_size, epochs, xs, ex):
        self.batch_size = batch_size
        self.epochs = epochs
        self.xs = xs
        self.dir_gen(ex)


    def training(self, model):
        (x_train, y_train), (x_test, y_test) = self.xs.get_train_test()
        x_train = x_train.reshape(-1, 44 * 44)
        x_test = x_test.reshape(-1, 44 * 44)
        y_train = keras.utils.to_categorical(y_train, 2)
        y_test = keras.utils.to_categorical(y_test, 2)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        print(x_train.shape, 'train samples', y_train.shape)
        print(x_test.shape, 'test samples', y_test.shape)

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=2,
                            validation_data=(x_test, y_test)
                            )

        # testing model
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # saving model
        self.book_keeping(history.history, model)

