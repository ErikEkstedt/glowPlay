import torch
import torch.nn as nn
import torch.optim as optim

from args import get_args, print_args
from train import train, calc_loss, sample_data, calc_z_shapes
from model import Glow


args = get_args()
print_args(args)


# Model
model_single = Glow(3,
        args.n_flow,
        args.n_block,
        affine=args.affine,
        conv_lu=not args.no_lu)
model = nn.DataParallel(model_single)
model = model.to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

train(args, model, optimizer)
