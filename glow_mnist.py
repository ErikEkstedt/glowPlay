import matplotlib.pyplot as plt
import torch

from ml_reusable.datasets.vision.mnist import load_mnist
from train import calc_z_shapes, calc_loss
from model import Glow


train_loader, test_loader = load_mnist()

n_flow = 16
n_block = 2
affine = True
img_size = 28
n_sample = 2
temp = 0.7
n_bits = 6
n_bins = 2. ** n_bits
img_channels = 1
model = Glow(img_channels, n_flow, n_block, affine=affine)

z_sample = []
z_shapes = calc_z_shapes(img_channels, img_size, n_flow, n_block)
for z in z_shapes:
    z_new = torch.randn(n_sample, *z) * temp
    z_sample.append(z_new)


for image, _ in train_loader:
    break

log_p, logdet, total_out = model(image + torch.rand_like(image) / n_bins)
loss, log_p, log_det = calc_loss(log_p, logdet, img_size, img_channels, n_bins)


with torch.no_grad():
    img = model.reverse(total_out).cpu().data


plt.subplot(2,1,1)
plt.title('Model image')
plt.imshow(img[0][0], aspect='auto')
plt.colorbar()
plt.subplot(2,1,2)
plt.title('Actual image')
plt.imshow(image[0][0], aspect='auto')
plt.colorbar()
plt.show()
