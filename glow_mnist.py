import matplotlib.pyplot as plt
import torch
from torch.optim import Adam

import numpy as np
from ml_reusable.datasets.vision.mnist import load_mnist
from train import calc_z_shapes, calc_loss
from model import Glow

from tqdm import tqdm
from tensorboardX import SummaryWriter


def get_fig(img):
    plt.close()
    fig = plt.figure()
    plt.subplot(2,2,1)
    plt.title('Model image')
    plt.imshow(img[0][0].cpu(), aspect='auto')
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(img[1][0].cpu(), aspect='auto')
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(img[2][0].cpu(), aspect='auto')
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(img[3][0].cpu(), aspect='auto')
    plt.colorbar()
    return fig


writer = SummaryWriter()


train_loader, test_loader = load_mnist()


n_flow = 10
n_block = 2
affine = True
img_size = 28
n_sample = 4
temp = 0.7
n_bits = 5
n_bins = 2. ** n_bits
img_channels = 1

model = Glow(img_channels, n_flow, n_block, affine=affine)


z_sample = []
z_shapes = calc_z_shapes(img_channels, img_size, n_flow, n_block)
for z in z_shapes:
    z_new = torch.randn(n_sample, *z) * temp
    z_sample.append(z_new.to('cuda'))


model.to('cuda')
optimizer = Adam(model.parameters(), lr=1e-4)

plot = False
i = 0
total_loss = []

for i in range(100):
    for image, _ in tqdm(train_loader):
        optimizer.zero_grad()
        image = image.to('cuda')
        log_p, logdet, out = model(image + torch.rand_like(image) / n_bins)
        loss, log_p, log_det = calc_loss(log_p, logdet, img_size, img_channels, n_bins)
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss', loss.cpu().item(), i)
        i += 1
        total_loss.append(loss.cpu().item())
        break
        if plot:
            plt.scatter(i, loss.cpu().item(), color='blue')
            plt.pause(0.0001)

    with torch.no_grad():
        img = model.reverse(z_sample)

    fig = get_fig(img)
    writer.add_figure('fixed sample Z', fig)

    with torch.no_grad():
        log_p, logdet, total_out = model(image)
        img = model.reverse(total_out).cpu().data

    fig = get_fig(img[:4])
    writer.add_figure('Inverted', fig)

    fig = get_fig(image[:4])
    writer.add_figure('Original', fig)

# Invertible ?

# with torch.no_grad():
#     log_p, logdet, total_out = model(image.to('cuda'))
#     img = model.reverse(total_out).cpu().data
#     # img = model.reverse(z_shapes).cpu().data

# plt.subplot(2,1,1)
# plt.title('Model image')
# # plt.imshow(torch.sigmoid(img[0][0]), aspect='auto')
# plt.imshow(img[0][0].cpu(), aspect='auto')
# plt.colorbar()
# plt.subplot(2,1,2)
# plt.title('Actual image')
# plt.imshow(image[0][0].cpu(), aspect='auto')
# plt.colorbar()
# plt.show()


# Sampling


