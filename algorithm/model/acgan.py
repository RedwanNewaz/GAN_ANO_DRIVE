import keras.backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
K.set_image_dim_ordering('th')
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
        aux = Dense(self.num_classes, activation='softmax', name='auxiliary')(features)
        D=Model(input=chunk, output=[fake, aux])

        D.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
                      optimizer=Adam(1e-4, decay=1e-4),
                      metrics=['accuracy'])

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
                              activation='tanh', init='glorot_normal'))

        # upsample to (..., 44, 44)
        cnn.add(UpSampling2D(size=(2, 2)))
        cnn.add(Convolution2D(128, 5, 5, border_mode='same',
                              activation='tanh', init='glorot_normal'))
        # take a channel axis reduction
        cnn.add(Convolution2D(1, 2, 2, border_mode='same',
                              activation='tanh', init='glorot_normal'))
        # this is the z space commonly refered to in GAN papers
        latent = Input(shape=(self.latent_size,))
        # this will be our label
        chunk_class = Input(shape=(1,), dtype='int32')
        # 2 classes in dataset
        cls = Flatten()(Embedding(self.num_classes, self.latent_size,
                                  init='glorot_normal')(chunk_class))
        h = merge([latent, cls], mode='mul')
        fake_image = cnn(h)
        G=Model(input=[latent, chunk_class], output=fake_image)

        G.compile(loss='categorical_crossentropy',
                      optimizer=Adam(1e-4, decay=1e-4),
                      metrics=['accuracy'])

        return G

class gan_model(object):
    def __init__(self):
        G_net=Generator()
        D_net=Discriminator()
        self.latent_size =100
        self.G=G_net.build_generator()
        self.D=D_net.build_discriminator()

    def get_model(self):
        latent = Input(shape=(self.latent_size,))
        chunk_class = Input(shape=(1,), dtype='int32')
        fake = self.G([latent, chunk_class])
        self.D.trainable = False
        fake, aux = self.D(fake)
        self.gan = Model(input=[latent, chunk_class], output=[fake, aux])
        self.gan.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
                  optimizer=Adam(1e-4, decay=1e-4),
                  metrics=['accuracy'])

        return self.G,self.D,self.gan



