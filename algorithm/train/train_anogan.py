from collections import defaultdict
import numpy as np
from keras.utils.generic_utils import Progbar
from algorithm.train.base_train import base_train


class train(base_train):
    def __init__(self, batch_size, epochs, xs, ex):
        self.batch_size = batch_size
        self.epochs = epochs
        self.xs = xs
        self.dir_gen(ex)
        self.latent_size=100

    def training(self, model):
        (x_train, y_train), (x_test, y_test) = self.xs.get_train_test()
        X_train = x_train.reshape(-1, 44, 44,1)
        self.G, self.D, self.GAN = model


        self.train_history = defaultdict(list)

        for epoch in range(self.epochs):
            print('Epoch {} of {}'.format(epoch + 1, self.epochs))

            n_iter = int(X_train.shape[0] / self.batch_size)
            progress_bar = Progbar(target=n_iter)

            epoch_gen_loss = []
            epoch_disc_loss = []

            for idx in range(n_iter):
                progress_bar.update(idx, force=True)

                noise = np.random.uniform(-1, 1, size=(self.batch_size, self.latent_size))
                image_batch = X_train[idx * self.batch_size:(idx + 1) * self.batch_size]
                generated_images = self.G.predict(noise, verbose=2)
                X = np.concatenate((image_batch, generated_images))
                y = np.array([1] * self.batch_size + [0] * self.batch_size)
                d_loss = self.D.train_on_batch(X, y)
                epoch_disc_loss.append(d_loss)
                noise = np.random.uniform(-1, 1, (2 * self.batch_size, self.latent_size))
                self.D.trainable = False
                # we want to train the G to trick the D
                # For the G, we want all the {fake, not-fake}
                # labels to say not-fake
                g_loss = self.GAN.train_on_batch(noise, np.ones(2 * self.batch_size))
                self.D.trainable = True
                epoch_gen_loss.append(g_loss)

            D_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
            G_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

            # generate an epoch report on performance
            self.train_history['G'].append(G_train_loss)
            self.train_history['D'].append(D_train_loss)

            ROW_FMT = '{0:<22s} | {1:<15.3f}'
            print('\n{0:<22s} | {1:15s}'.format('component', 'loss'))
            print('-' * 30)
            print(ROW_FMT.format('G (train)', self.train_history['G'][-1]))
            print(ROW_FMT.format('D (train)', self.train_history['D'][-1]))

        # saving model
        self.book_keeping(self.train_history,[self.G,self.D,self.GAN])

