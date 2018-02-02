import os
import pandas as pd


class base_train(object):
    gan_training=False


    def dir_gen(self, ex):
        file_path = 'result/{}/{}'.format(ex.model,ex.version)
        if(not os.path.exists(file_path)):
            os.makedirs(file_path)
            print('new dir: ',file_path)
        else:
            print('exits dir: ',file_path)

        if (ex.adversarial_train):
            self.gan_training=True
            self.weight_name = [file_path + '{}_weight_{}_{}.hdf5'.format(ex.model, name, ex.version) for name in
                                ['G', 'D', 'GAN']]
            self.model_name = [file_path + '{}_model_{}_{}.h5'.format(ex.model,name, ex.version) for name in ['G', 'D', 'GAN']]
            self.history_name = file_path + '{}_history_{}.csv'.format(ex.model,ex.version)
        else:
            self.weight_name = file_path + '{}_weight_{}.hdf5'.format(ex.model,ex.version)
            self.model_name = file_path + '{}_model_{}.h5'.format(ex.model,ex.version)
            self.history_name = file_path + '{}_history_{}.csv'.format(ex.model,ex.version)

    def book_keeping(self,history,model):
        if(self.gan_training):
            G,D,GAN=model
            # saving model
            for i, m in enumerate([G, D, GAN]):
                m.save_weights(self.weight_name[i])
                m.save(self.model_name[i])
        else:
            model.save_weights(self.weight_name)
            model.save(self.model_name)

        # saving history
        df = pd.DataFrame(history)
        df.to_csv(self.history_name)


