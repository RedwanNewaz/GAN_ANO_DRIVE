from keras.layers import Dense, Flatten, LeakyReLU, Input, Activation,Reshape
from keras.layers import BatchNormalization
from keras.regularizers import l1_l2
from keras.layers.convolutional import Convolution2D
from keras.models import Model,Sequential
from keras.layers.convolutional import UpSampling2D,MaxPooling2D,AveragePooling2D
from keras.optimizers import RMSprop
import keras.backend as K
K.set_image_dim_ordering('th')
reg=1e-5
class Discriminator(object):
    def __init__(self):
        self.x_dim = (1, 1, 44,44)
        self.name = 'wgan/d_net'
        self.reg = lambda: l1_l2(l1=reg, l2=reg)
        self.h = 5 * 1
        self.x = 32 * 4
        self.channel = 1
        self.h_dim = [64, 128, 256]

    def build_discriminator(self):
        model = Sequential()
        for i, y in enumerate(self.h_dim):
            if i == 0:# FIRST LAYER
                model.add(Convolution2D(y, self.h, self.h, border_mode='same', W_regularizer=self.reg(),
                                        input_shape=( 1, 44, 44)))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(LeakyReLU(0.2))
            else:# INTERMEDIATE LAYER
                model.add(Convolution2D(y, self.h, self.h, border_mode='same', W_regularizer=self.reg()))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(LeakyReLU(0.2))

        else:
            model.add(Convolution2D(1, self.h, self.h, border_mode='same', W_regularizer=self.reg()))
            model.add(AveragePooling2D(pool_size=(4, 4), border_mode='valid'))
            model.add(Flatten())
            model.add(Activation('sigmoid'))
        return model

class Generator(object):
    def __init__(self):
        self.z_dim = 100
        self.x_dim = (1, 1, 44,44)
        self.name = 'wgan/g_net'
        self.reg = lambda: l1_l2(l1=reg, l2=reg)
        self.h = 5 * 1
        self.x = 11
        self.channel = 1

        self.h_dim = [int(256), int(128), int(64)]

    def build_generator(self):
        model = Sequential()
        for i, y in enumerate(self.h_dim):
            if i == 0:# FIRST LAYER
                model.add(Dense(y * self.x * self.x, input_dim=100, W_regularizer=self.reg()))
                model.add(BatchNormalization(mode=0))
                model.add(Reshape((y, self.x, self.x)))
            elif i == len(self.h_dim): # BEFORE LAST LAYER
                model.add(Convolution2D(y,self.h, self.h, border_mode='same', W_regularizer=self.reg()))
            else: # INTERMEDIATE LAYER
                model.add(Convolution2D(y,self.h, self.h, border_mode='same', W_regularizer=self.reg()))
                model.add(BatchNormalization(mode=0, axis=1))
                model.add(LeakyReLU(0.2))
                model.add(UpSampling2D(size=(2, 2)))
        else: # LAST LAYER
            model.add(Convolution2D(self.channel ,self.h, self.h, border_mode='same', W_regularizer=self.reg()))
            model.add(Activation('sigmoid'))

        return model

class gan_model(object):
    def __init__(self):
        G_net=Generator()
        D_net=Discriminator()
        self.latent_size =100
        self.G=G_net.build_generator()
        self.D=D_net.build_discriminator()

    def get_model(self):
        ngpu=2
        gan_input = Input(shape=[self.latent_size ])
        H = self.G(gan_input)
        gan_output = self.D(H)

        self.gan = Model(gan_input, gan_output)

        # EX4 optimizer changed to RMSprop and binary cross entropy from Adam and mse

        opt=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        self.D.compile(loss='binary_crossentropy', optimizer=opt)
        self.G.compile(loss='binary_crossentropy', optimizer=opt)
        self.gan.compile(loss='binary_crossentropy', optimizer=opt)



        return self.G,self.D,self.gan


if __name__ == '__main__':
    G=Generator()
    G=G.build_generator()
    G.summary()

    D=Discriminator()
    D=D.build_discriminator()
    D.summary()
    gan=gan_model()
    _,_,gan=gan.get_model()
    gan.summary()
