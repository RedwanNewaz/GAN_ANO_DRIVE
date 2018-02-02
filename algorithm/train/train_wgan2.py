import time
import numpy as np
from keras.utils.generic_utils import Progbar
from collections import defaultdict
from algorithm.train.base_train import base_train


class train(base_train):
    def __init__(self, batch_size, epochs, xs, ex):
        self.batch_size = batch_size
        self.epochs = epochs
        self.xs = xs
        self.dir_gen(ex)
        self.latent_size = 100
        self.get_next_batch(config=True)


    def get_next_batch(self, config=True):
        if config is True:
            (x_train, y_train), (_, _) = self.xs.get_train_test()
            # poistive class
            self.x_train = x_train[y_train == 1]
            self.nb_batches = int(self.x_train.shape[0] / self.batch_size)
            self.index = -1  # otherwise it avoids first batch when training start
        else:
            self.index = self.index + 1 if self.index < self.nb_batches - 1 else 0
            return self.x_train[self.index * self.batch_size:(self.index + 1) * self.batch_size]

    def get_train_pair(self, batch, G, type):
        X, y = None, None
        if type == 'discriminator':
            x = self.get_next_batch(batch)
            x = x.reshape((x.shape[0], 1, 44, 44))
            batch_size = x.shape[0]
            z = np.random.uniform(-1, 1, (batch_size, self.latent_size))
            x_ = G.predict(z, batch_size=batch_size)

            # s=lambda x:np.shape(x)
            # print(s(x_),s(x))

            X = np.concatenate((x, x_))
            y = np.zeros([2 * batch_size, 1])
            y[0:batch_size, 0] = 1

        elif type == 'gan':
            X = np.random.uniform(-1, 1, (self.batch_size, self.latent_size))
            y = np.ones([self.batch_size, 1])

        return X, y

    def pretrain_discriminator(self,batch,G,D):
        X, y = self.get_train_pair(batch, G,type='discriminator')
        try:
            d_loss = D.train_on_batch(X, y)
            print('pre training dloss',d_loss)
        except:
            print('Pre Training failed')
            raise

    def training(self, model):

        start_time = time.time()
        history = defaultdict(list)
        progress_bar = Progbar(target=100)
        G, D, GAN = model
        self.pretrain_discriminator(self.batch_size, G, D)

        def clip_d_weights():
            weights = [np.clip(w, -0.01, 0.01) for w in D.get_weights()]
            D.set_weights(weights)

        for t in range(self.epochs):
            d_iters = 5
            progress_bar.update(t % 100)
            if t % 500 == 0 or t < 25:
                d_iters = 100

            D.trainable = True
            # for t in tqdm(range(0, d_iters)):
            if d_iters > 5:
                for t in range(0, d_iters):
                    clip_d_weights()
                    X, y = self.get_train_pair(self.batch_size, G, type='discriminator')
                    # print GAN.input
                    d_loss= D.train_on_batch(X, y)

            # train Generator-Discriminator stack on input noise to non-generated output class
            # print('training pair gan')
            X, y = self.get_train_pair(self.batch_size, G, type='gan')

            D.trainable = False
            # print('training gan')
            g_loss = GAN.train_on_batch(X, y)

            if t % 100 == 0 or t < 100:
                # Train discriminator
                D.trainable = True
                X, y = self.get_train_pair(self.batch_size, G, type='discriminator')
                d_loss = D.train_on_batch(X, y)

                # Train Generator-Discriminator with discriminator fixed
                D.trainable = False
                X, y = self.get_train_pair(self.batch_size, G, type='gan')
                g_loss = GAN.train_on_batch(X, y)

                history['d_loss'].append(d_loss)
                history['g_loss'].append(g_loss)

                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                      (t + 1, time.time() - start_time, d_loss - g_loss, g_loss))

        # saving model
        self.book_keeping(history, [G, D, GAN])


