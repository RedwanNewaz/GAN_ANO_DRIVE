from keras.layers import Dense, Flatten, LeakyReLU, Input,Reshape, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Model,Sequential
from keras.layers.convolutional import UpSampling2D
from keras.optimizers import RMSprop
import keras.backend as K
K.set_image_dim_ordering('th')

reg=1e-5
class Discriminator(object):
    def __init__(self):
        self.ngpu=2
        self.num_classes=2

    def build_discriminator(self):
        cnn = Sequential()

        cnn.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2),
                              input_shape=(1, 44, 44)))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.5))

        cnn.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.5))

        cnn.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(2, 2)))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.5))

        cnn.add(Convolution2D(256, 3, 3, border_mode='same', subsample=(1, 1)))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.5))

        cnn.add(Flatten())

        chunk = Input(shape=(1, 44, 44))
        features = cnn(chunk)

        fake = Dense(1, activation='sigmoid', name='generation')(features)
        D=Model(input=[chunk], output=[fake])


        return D

class Generator(object):
    def __init__(self):
        self.latent_size=100
        self.num_classes=2

    def build_generator(self):
        cnn = Sequential()
        cnn.add(Dense(128 * 11 * 11, input_dim=self.latent_size, activation='tanh'))
        cnn.add(Reshape((128, 11, 11)))

        # upsample to (..., 22, 22)
        cnn.add(UpSampling2D(size=(2, 2)))
        cnn.add(Convolution2D(256, 5, 5, border_mode='same',
                              activation='tanh'))

        # upsample to (..., 44, 44)
        cnn.add(UpSampling2D(size=(2, 2)))
        cnn.add(Convolution2D(128, 5, 5, border_mode='same',
                              activation='tanh'))
        # take a channel axis reduction
        cnn.add(Convolution2D(1, 2, 2, border_mode='same',
                              activation='tanh'))
        # this is the z space commonly refered to in GAN papers
        latent = Input(shape=(self.latent_size,))


        fake_image = cnn(latent)
        G=Model(input=[latent], output=[fake_image])
        return G


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
