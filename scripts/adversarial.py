import torch.nn
import torch.nn.functional as F

GAN_TYPES = ['simple_gan', 'gan', 'wgan', 'wgan-gp']


def generator_loss(fake, gan_type='gan'):
    """

    :param fake: prediction from discriminator on fake generated data
    :param gan_type: type of gan that we are using [simple_gan,gan,wgan, wgan-gp]
    :return: generator loss
    """
    assert gan_type in GAN_TYPES, 'unknown GAN type: {}'.format(gan_type)

    if gan_type == 'simple_gan':
        return torch.mean(F.logsigmoid(1 - fake)).to(fake.device)

    elif gan_type == 'gan':
        return -torch.mean(F.logsigmoid(fake)).to(fake.device)

    else:
        assert gan_type in ['wgan', 'wgan_gp']
        return -torch.mean(fake).to(fake.device)


def discriminator_loss(fake, real, gan_type='gan', penalty=None) :
    """

    :param fake: prediction from discriminator on fake generated data
    :param real: prediction from discriminator on real data
    :param gan_type: type of gan that we are using [simple_gan,gan,wgan, wgan-gp]
    :param penalty: for wgan - gp
    :return: discriminator loss
    """
    assert gan_type in GAN_TYPES, 'unknown GAN type: {}'.format(gan_type)

    if gan_type in ['simple_gan', 'gan']:
        return - torch.mean(F.logsigmoid(real)).to(fake.device) - torch.mean(F.logsigmoid(-fake)).to(fake.device)

    else:
        assert gan_type in ['wgan', 'wgan_gp']
        if None is penalty:
            return -torch.mean(real).to(fake.device) + torch.mean(fake).to(fake.device)

        else:
            # TODO Vage penalty (fit_adversarial-64_4-wgan-Copy1)
            return -torch.mean(real).to(fake.device) + torch.mean(fake).to(fake.device) + penalty.to(fake.device)
