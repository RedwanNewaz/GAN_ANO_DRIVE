from keras.layers import Dense, Flatten, LeakyReLU, Input, Activation,Reshape, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import SGD
from keras.models import Model
import keras.backend as K
K.set_image_data_format('channels_last')
reg=1e-5
class Discriminator(object):
    def __init__(self):
        self.n_filters = 64
        self.input_shape = (44, 44, 1)

    def build_discriminator(self):

        disc_input = Input(shape=self.input_shape)

        cnn = Conv2D(self.n_filters, 5, padding='same')(disc_input)
        cnn = LeakyReLU()(cnn)
        cnn = MaxPooling2D(pool_size=(2, 2))(cnn)

        cnn = Conv2D(self.n_filters * 2, 5, padding='same')(cnn)
        cnn = LeakyReLU()(cnn)
        cnn = MaxPooling2D(pool_size=(2, 2))(cnn)

        cnn = Flatten()(cnn)

        cnn = Dense(1024)(cnn)
        cnn = LeakyReLU()(cnn)

        cnn = Dense(1)(cnn)
        disc_output = Activation('sigmoid')(cnn)


        return Model(disc_input, disc_output)




class Generator(object):
    def __init__(self):
        self.latent_size=100
        self.n_filters = 64

    def build_generator(self):


        # generator input of latent space vector Z, typically a 1D vector
        gen_input = Input(shape=(self.latent_size,))
        # layer 1 - higher dimensional dense layer
        cnn = Dense(1024)(gen_input)
        cnn = Activation('relu')(cnn)
        # layer 2 - higer dimensional dense layer
        cnn = Dense(11 * 11 * 128)(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Activation('relu')(cnn)
        # transform 1D to 3D matrix (2D image plus channels)
        cnn = Reshape((11, 11, 128))(cnn)
        # layer 3 - convulational layer - filter matching
        cnn = UpSampling2D(size=(2, 2))(cnn)
        cnn = Conv2D(self.n_filters, 5, padding='same')(cnn)
        cnn = Activation('relu')(cnn)
        # layer 4 - convulational layer - channel reducer
        cnn = UpSampling2D(size=(2, 2))(cnn)
        cnn = Conv2D(1, 5, padding='same')(cnn)
        gen_output = Activation('tanh')(cnn)


        return Model(gen_input, gen_output)


class gan_model(object):
    def __init__(self):
        G_net=Generator()
        D_net=Discriminator()
        self.latent_size =100
        self.G=G_net.build_generator()
        self.D=D_net.build_discriminator()

    def get_model(self):

        gan_input = Input(shape=[self.latent_size ])
        H = self.G(gan_input)
        self.D.trainable = False
        gan_output = self.D(H)

        self.gan = Model(gan_input, gan_output)


        # EX4 optimizer changed to RMSprop and binary cross entropy from Adam and mse

        d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        e2e_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)


        self.G.compile(loss='binary_crossentropy', optimizer="SGD")
        self.gan.compile(loss='binary_crossentropy', optimizer=e2e_optim)
        self.D.trainable = True
        self.D.compile(loss='binary_crossentropy', optimizer=d_optim)



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
