import argparse
import torch


def print_args(args):
    for k, v in vars(args).items():
        print(f'{k}: {v}')


def get_args():
    parser = argparse.ArgumentParser(description='Glow trainer')
    parser.add_argument('--batch', default=16, type=int, help='batch size')
    parser.add_argument('--iter', default=200000, type=int, help='maximum iterations')
    parser.add_argument('--n_flow', default=32, type=int, help='number of flows in each block')
    parser.add_argument('--n_block', default=4, type=int, help='number of blocks')
    parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
    parser.add_argument('--cpu', action='store_true', help='force cpu')
    parser.add_argument('--affine', action='store_true', help='use affine coupling instead of additive')
    parser.add_argument('--n_bits', default=5, type=int, help='number of bits')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--img_size', default=64, type=int, help='image size')
    # parser.add_argument('--device_ids', default=64, type=int, help='image size')
    parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
    parser.add_argument('--n_sample', default=20, type=int, help='number of samples')
    parser.add_argument('path', metavar='PATH', type=str, help='Path to image directory')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = 'cpu' if args.cpu else device
    return args
