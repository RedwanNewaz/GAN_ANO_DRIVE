from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop,Adam,Adadelta
from keras import backend as K

K.set_image_dim_ordering('th')
class Discriminator(object):
    def __init__(self,xs,num_classes):
        self.xs=xs
        self.ngpu=2
        self.num_classes=num_classes

    def get_model(self):
        img_rows, img_cols = 44, 44
        input_shape = (1, img_rows, img_cols)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='tanh',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adadelta(),
                      metrics=['accuracy'])
        return model