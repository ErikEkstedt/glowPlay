import torch
import torch.nn as nn
import torch.nn.functional as F


############################## Utils ##########################


def channel_mean_std(x):
    idx = [f for f in range(x.dim()) if f != 1]
    return x.mean(dim=idx)

def channel_mean_var(x):
    x = x.transpose(1, 0)  # switch channel and batch dim
    x = x.flatten(1)
    return x.mean(dim=1), x.var(dim=1)


############################## Modules ##########################


class ActNorm(nn.Module):
    """
    Activation Normalization

    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance.
    After initialization, `bias` and `scale` will be trained as parameters.
    """

    def __init__(self, num_channels, data_dim=1):
        super().__init__()
        self.num_channels = num_channels
        self.initialized = False
        self.data_dim = data_dim

        # Parameters
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.log_scale = nn.Parameter(torch.zeros(num_channels))

        self.register_parameter("bias", self.bias) 
        self.register_parameter("log_scale", self.log_scale) 

    def initialize(self, input):
        '''
        The scale and bias are initialized such that the mini-batch after this
        initial normalization the minibatch will have zero mean and unit
        variance over the channel dimensions.
        '''
        self.initialized = True
        with torch.no_grad():
            '''Find channel mean and variance'''
            bias, var = channel_mean_var(input) 
            self.bias.data.copy_(-bias) # channel dim
            self.log_scale.data.copy_(-torch.log(1 / (var + 1e-6)))

        for i in range(self.data_dim):
            self.bias.data = self.bias.unsqueeze(-1)
            self.log_scale.data = self.log_scale.unsqueeze(-1)

    def encode(self, x, log_det=None):
        ''' not reverse
        f^-1: X -> Z
        Shift then Scale
        '''
        assert self.initialized
        x_shifted = x + self.bias 
        z = x_shifted * torch.exp(self.log_scale) 
        return z, log_det

    def decode(self, z, log_det=None):
        ''' reverse
        f: Z -> X
        Scale then shift
        '''
        assert self.initialized
        z = z * torch.exp(-self.log_scale)
        x = z - self.bias
        return x, log_det


############################## Glow ##############################

class Glow(nn.Module):
    def __init__(
            self,
            K=11,  # Steps of flow
            L=3):  # Multi-scale layers
        super().__init__()

    def forward(self, x):
        return x


class Step_of_Flow(nn.Module):
    def __init__(self):
        self.actnorm = Actnorm()
        self.permutation = Invertible_1x1_conv()

    def encode(self, x):
        pass

    def decode(self, z):
        pass


class Level(nn.Module):
    def __init__(self):
        super(Level, self).__init__()

        self.squeeze = Squeeze()
        self.flow = Step_of_Flow()
        self.split = Split()

    def forward(self, x):
        return x


if __name__ == "__main__":

    channel_shape = (32, 1, 10)
    a = torch.ones(channel_shape) + torch.randn(channel_shape)*0.1
    b = torch.ones(channel_shape)*2 + torch.randn(channel_shape)*0.2
    c = torch.ones(channel_shape)*3 + torch.randn(channel_shape)*0.3
    mini_batch = torch.cat((a,b,c), dim=1)
    data_dim = mini_batch.dim() - 2

    actnorm = ActNorm(mini_batch.shape[1], data_dim)
    actnorm.initialize(mini_batch)
    out, _ = actnorm.encode(mini_batch)

    pred_mini_batch, _ = actnorm.decode(out)


    actnorm = test_actnorm()
