import torch
from custom_glow import ActNorm, channel_mean_var 


def test_actnorm(mini_batch=None):
    # mini_batch = torch.randn((32, 140))  # face
    if mini_batch is None: 
        channel_shape = (32, 1, 20)
        a = torch.ones(channel_shape) + torch.randn(channel_shape)*0.1
        b = torch.ones(channel_shape)*2 + torch.randn(channel_shape)*0.2
        c = torch.ones(channel_shape)*3 + torch.randn(channel_shape)*0.3
        mini_batch = torch.cat((a,b,c), dim=1)

    actnorm = ActNorm(num_channels=mini_batch.shape[1])
    print('Before init')
    print('Log scale: ', actnorm.log_scale.data)
    print('bias: ', actnorm.bias.data)

    actnorm.initialize(mini_batch)
    print('After init')
    print('Log scale: ', actnorm.log_scale.data)
    print('bias: ', actnorm.bias.data, '\n')
    print('-'*50)

    mean, var = channel_mean_var(mini_batch)
    print('Before actnorm')
    print('mini_batch: ', mini_batch.shape)
    print('mean: ', mean)
    print('std: ', var)

    out, _ = actnorm.encode(mini_batch, None)
    mean, var = channel_mean_var(out)
    print('After actnorm')
    print('out: ', out.shape)
    print('mean: ', mean)
    print('std: ', var)

    mini_batch_, _ = actnorm.decode(out)
    print('reverse actnorm')
    print('out: ', out.shape)
    print('mini_batch_: ', mini_batch_.shape)
    print('mean: ', mean)
    print('std: ', var)
    print('mini_batch - mini_batch_ = 0 ?', torch.mean(mini_batch - mini_batch_))
    return actnorm
