from algorithm.train.base_train import base_train
import numpy as np
from keras.utils.generic_utils import Progbar
from collections import defaultdict


class train(base_train):
    def __init__(self, batch_size, epochs, xs, ex):
        self.batch_size = batch_size
        self.epochs = epochs
        self.xs = xs
        self.dir_gen(ex)



    def training(self, model):
        (x_train, y_train), (x_test, y_test) = self.xs.get_train_test()

        x_train = x_train.reshape(-1,1, 44, 44)
        x_test = x_test.reshape(-1, 1, 44 , 44)


        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        print(x_train.shape, 'train samples', y_train.shape)
        print(x_test.shape, 'test samples', y_test.shape)
        #-------------------------------------------------------------
        G,D,GAN=model
        nb_batches = int(x_train.shape[0] / self.batch_size)
        progress_bar = Progbar(target=nb_batches)
        latent_size=100
        num_classes=2
        history = defaultdict(list)
        batch_size=self.batch_size
        for epoch in range(self.epochs):
            epoch_gen_loss = []
            epoch_disc_loss = []
            for index in range(nb_batches):
                progress_bar.update(index)
                noise = np.random.uniform(-1, 1, (batch_size, latent_size))
                image_batch = x_train[index * batch_size:(index + 1) * batch_size]
                label_batch = y_train[index * batch_size:(index + 1) * batch_size]
                sampled_labels = np.random.randint(0, num_classes, batch_size)
                # training discriminator
                generated_images = G.predict(
                    [noise, sampled_labels.reshape((-1, 1))], verbose=0)
                X = np.concatenate((image_batch, generated_images))
                y = np.array([1] * batch_size + [0] * batch_size)
                aux_y = np.concatenate((label_batch, sampled_labels), axis=0)
                epoch_disc_loss.append(D.train_on_batch(X, [y, aux_y]))
                # trianing GAN model
                noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
                sampled_labels = np.random.randint(0, num_classes, 2 * batch_size)
                trick = np.ones(2 * batch_size)
                epoch_gen_loss.append(GAN.train_on_batch(
                    [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))

            d_loss = np.mean(np.array(epoch_disc_loss), axis=0)
            g_loss = np.mean(np.array(epoch_gen_loss), axis=0)
            history['g_loss'].append(np.sum(g_loss, axis=0))
            history['d_loss'].append(np.sum(d_loss, axis=0))
            print(' Epoch {} of {}'.format(epoch + 1, self.epochs))
            print(' g_loss {:3.3f} d_loss {:3.3f}'.format(np.sum(g_loss,axis=0),np.sum(d_loss,axis=0)))

        self.book_keeping(history,[G,D,GAN])





