import argparse
import importlib

model_dict={
    "acgan":'algorithm.model.acgan',
    "anogan": 'algorithm.model.anogan',
    "cnn": 'algorithm.model.cnn',
    "mlp": 'algorithm.model.mlp',
    "wgan": 'algorithm.model.wgan',
    "wgan2": 'algorithm.model.wgan2'

}
train_dict={
    "acgan":'algorithm.train.train_acgan',
    "anogan": 'algorithm.train.train_anogan',
    "cnn": 'algorithm.train.train_cnn',
    "mlp": 'algorithm.train.train_mlp',
    "wgan": 'algorithm.train.train_wgan',
    "wgan2": 'algorithm.train.train_wgan2'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Learn_Lane_Change_Behavior')
    parser.add_argument('--dataloader', type=str, default='algorithm')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--adversarial_train', type=bool, default=False)
    parser.add_argument('--version', type=str, default='0.1')
    parser.add_argument('--ngpu', type=int, default=2)
    args = parser.parse_args()

    data = importlib.import_module(args.dataloader)
    model = importlib.import_module(model_dict[args.model])
    train = importlib.import_module(train_dict[args.model])

    xs = data.DataSampler(chunk=44, percent=0.7)
    d_net = model.Discriminator(xs=xs, num_classes=2) if args.adversarial_train is False else model.gan_model()

    train_model=train.train(batch_size=10, epochs=50, xs=xs, ex=args)
    train_model.training(d_net.get_model())

