# GAN_ANO_DRIVE

In this repository, I have will show you how to use GAN to learn lane changing behavior of drivers.
My objective is to obtain the optimal decision boundary of GANs with limited amount of training dataset.

## Hypothesis
I make two hypotheses in order to achieve my goal
* Using one class of training data we can achieve optimal decision boundary.*
* Discriminator of GANs can generalize the data well.

## Validation

* Active Test: where the discriminator can directly provide the class labels, e.g. CNN, MLP, ACGAN
* Passive Test: where the class label is computed either based on the indvidual prediction probabilities of GANs components or combined prediction probabilities of GANs components. 

![](https://github.com/RedwanNewaz/GAN_ANO_DRIVE/blob/master/fig/gan_architecture.png)

## Dataset generation
![](https://github.com/RedwanNewaz/GAN_ANO_DRIVE/blob/master/fig/dataset.png)
![](https://github.com/RedwanNewaz/GAN_ANO_DRIVE/blob/master/fig/likert.png)

## Conclusions
These conclusions are based on the risky lane changing behavior from NU9600 dataset. The dataset is provided by the Takeda laboratory at Nagoya University, Japan. 

* F1-score, Recall for the passive test are 0 where for the active test are 1. It validates the well known fact that the discriminative models are more accurate to model the conditional density function P(C|X).
* Using generative model for the classification task works well only in case of VGAN architecture. Therefore, existing works use the generator of VGAN architecture to detect the anomalies.   
* The discriminator of ACGANs outperforms other discriminators since it was trained using both positive and negative examples. However, in case of the generative model, the classification result is worst. Therefore, we can say that we don’t need to improve the generative model when  positive and negative examples are available. 
In case of one class training, the discriminator of WGAN outperforms others.  


