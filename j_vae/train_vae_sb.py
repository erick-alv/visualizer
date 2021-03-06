from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import torch.utils.data
#from utils.os_utils import make_dir
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
#from j_vae.common_data import train_file_name, vae_sb_weights_file_name
this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'

#TODO verify why this version of code is not working anymore
def spatial_broadcast(z, width, height):
    z_b = np.tile(A=z, reps=(height, width, 1))
    x = np.linspace(-1,1, width)
    y = np.linspace(-1,1,width)
    x_b, y_b = np.meshgrid(x, y)
    x_b = np.expand_dims(x_b, axis=2)
    y_b = np.expand_dims(y_b, axis=2)
    z_sb = np.concatenate([z_b, x_b, y_b], axis=-1)
    return z_sb

def torch_spatial_broadcast(z, width, height, device):
    z_b = torch.stack([z] * width, dim=1)
    z_b = torch.stack([z_b] * height, dim=2)
    x = torch.linspace(-1, 1, steps=width)
    y = torch.linspace(-1, 1, steps=height)
    n = z.size()[0]
    x_b, y_b = torch.meshgrid(x, y)
    x_b = torch.unsqueeze(x_b, 2)
    y_b = torch.unsqueeze(y_b, 2)
    x_b = torch.stack([x_b] * n, dim=0).to(device)
    y_b = torch.stack([y_b] * n, dim=0).to(device)
    z_sb = torch.cat([z_b, x_b, y_b], dim=-1)
    z_sb = z_sb.permute([0, 3, 1, 2])
    return z_sb


class VAE_SB(nn.Module):
    def __init__(self, device, img_size, latent_size, full_connected_size=320, input_channels=3,
                 kernel_size=3, encoder_stride=2, decoder_stride=1, extra_layer=False):
        super(VAE_SB, self).__init__()
        self.device = device
        self.img_size = img_size
        self.extra_layer = extra_layer
        self.c1 = nn.Conv2d(in_channels=input_channels, kernel_size=kernel_size,
                            stride=encoder_stride, out_channels=img_size, padding=1)
        self.c2 = nn.Conv2d(in_channels=img_size, kernel_size=kernel_size,
                            stride=encoder_stride, out_channels=img_size, padding=0)
        self.c3 = nn.Conv2d(in_channels=img_size, kernel_size=kernel_size,
                            stride=encoder_stride, out_channels=img_size, padding=1)
        self.c4 = nn.Conv2d(in_channels=img_size, kernel_size=kernel_size,
                            stride=encoder_stride, out_channels=img_size, padding=1)
        #the number of cahnnels is img_size
        self.fc1 = nn.Linear(5 * 5 * img_size, full_connected_size)
        # Try to reduce
        self.fc21 = nn.Linear(full_connected_size, latent_size)
        self.fc22 = nn.Linear(full_connected_size, latent_size)

        self.dc1 = nn.Conv2d(in_channels=latent_size+2, kernel_size=kernel_size,
                             stride=decoder_stride, out_channels=img_size, padding=1)
        self.dc2 = nn.Conv2d(in_channels=img_size,
                             kernel_size=kernel_size, stride=decoder_stride, out_channels=img_size, padding=1)
        if extra_layer:
            self.dc3 = nn.Conv2d(in_channels=img_size,
                                 kernel_size=kernel_size, stride=decoder_stride, out_channels=img_size, padding=1)
            self.dc4 = nn.Conv2d(in_channels=img_size,
                                 kernel_size=kernel_size, stride=decoder_stride, out_channels=3, padding=1)
        else:
            self.dc3 = nn.Conv2d(in_channels=img_size,
                                 kernel_size=kernel_size, stride=decoder_stride, out_channels=3, padding=1)


    def encode(self, x):
        #h1 = F.relu(self.fc1(x))
        #return self.fc21(h1), self.fc22(h1)
        e1 = F.relu(self.c1(x))
        e2 = F.relu(self.c2(e1))
        e3 = F.relu(self.c3(e2))
        e4 = F.relu(self.c4(e3))
        e = e4.reshape(-1, 5 * 5 * self.img_size)
        e = F.relu(self.fc1(e))
        return self.fc21(e), self.fc22(e)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        #return (mu + eps * std) / 11
        return (mu + eps * std)
        #return mu

    # maybe z * 11
    def decode(self, z):
        tz = torch_spatial_broadcast(z, self.img_size, self.img_size, self.device)
        d1 =  F.relu(self.dc1(tz))
        d2 = F.relu(self.dc2(d1))
        d3 = F.relu(self.dc3(d2))
        if self.extra_layer:
            d4 = F.relu(self.dc4(d3))
            return torch.sigmoid(d4)
        else:
            return torch.sigmoid(d3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, beta):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    # Try to adjust
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta*KLD

# torch.Size([128, 1, img_size, img_size])
def train(epoch, model, optimizer, device, log_interval, train_file, batch_size, beta):
    model.train()
    train_loss = 0
    data_set = np.load(train_file)
    
    data_size = len(data_set)
    data_set = np.split(data_set, data_size / batch_size)

    for batch_idx, data in enumerate(data_set):
        data = torch.from_numpy(data).float().to(device)
        data /= 255
        data = data.permute([0, 3, 1, 2])
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), data_size,
                100. * (batch_idx+1) / len(data_set),
                loss.item() / len(data)))
            print('Loss: ', loss.item() / len(data))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / data_size))


def train_Vae(batch_size, img_size, latent_size, train_file, vae_weights_path, beta, epochs=100, no_cuda=False, seed=1,
              log_interval=100, load=False):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")
    if load:
        model = VAE_SB(device, img_size=img_size, latent_size=latent_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        checkpoint = torch.load(vae_weights_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        start_epoch = epoch + 1
    else:
        model = VAE_SB(device, img_size=img_size, latent_size=latent_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        start_epoch = 1

    for epoch in range(start_epoch, epochs + start_epoch):
        train(epoch=epoch, model=model, optimizer=optimizer, device=device, log_interval=log_interval,
              train_file=train_file, batch_size=batch_size, beta=beta)
        if not (epoch % 5) or epoch == 1:
            test_on_data_set(model, device,filename_suffix='epoch_{}'.format(epoch), latent_size=latent_size,
                             train_file=train_file)
            print('Saving Progress!')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, vae_weights_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, vae_weights_path+'_epoch_'+str(epoch))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, vae_weights_path)

def test_Vae(img_size, latent_size, train_file, vae_weights_path,no_cuda=False, seed=1):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")
    model = VAE_SB(device, img_size=img_size, latent_size=latent_size).to(device)
    checkpoint = torch.load(vae_weights_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    data_set = np.load(train_file)
    test_on_data_set(model, device, latent_size=latent_size, filename_suffix='test', data_set=data_set.copy())


def test_on_data_set(model, device, filename_suffix, latent_size, train_file):
    data_set = np.load(train_file)
    data_size = len(data_set)
    idx = np.random.randint(0, data_size, size=10)
    data = data_set[idx]
    with torch.no_grad():
        data = torch.from_numpy(data).float().to(device)
        data /= 255
        data = data.permute([0, 3, 1, 2])
        recon, mu, logvar = model(data)

        mu = mu.view(10, latent_size)
        logvar = logvar.view(10, latent_size)

        comparison = torch.cat([data, recon])
        save_image(comparison.cpu(), this_file_dir+'results/reconstruction_{}.png'.format(filename_suffix),
                   nrow=10)

def load_Vae(path, img_size, latent_size, no_cuda=False, seed=1, full_connected_size=320, input_channels=3,
                 kernel_size=3, encoder_stride=2, decoder_stride=1, extra_layer=False):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    model = VAE_SB(device, img_size=img_size, latent_size=latent_size, full_connected_size=full_connected_size,
                   input_channels=input_channels, kernel_size=kernel_size, encoder_stride=encoder_stride,
                   decoder_stride=decoder_stride, extra_layer=extra_layer).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model

#adapted from https://github.com/Natsu6767/Variational-Autoencoder/blob/master/main.py
import matplotlib.pyplot as plt
from scipy.stats import norm

def show_2d_manifold(img_size, vae_weights_path, no_cuda=False, seed=1):
    latent_size = 2
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")
    model = VAE_SB(device, img_size=img_size, latent_size=latent_size).to(device)
    checkpoint = torch.load(vae_weights_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    n = 20  # figure with nxn images
    figure = np.zeros((img_size * n, img_size * n, 3))
    # Contruct grid of latent variable values.
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n, endpoint=True))#with probabilities to values of the distribution
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n, endpoint=True))
    #grid_x = np.linspace(-2, 2, n, endpoint=True)#with values of the distribution
    #grid_y = np.linspace(-2, 2, n, endpoint=True)
    # Decode for each square in the grid.
    for i, xi in enumerate(grid_x):
        for j, yj in enumerate(grid_y):
            z_sample = np.array([xi, yj])
            z_sample = torch.from_numpy(z_sample).to(device).float()
            z_sample = torch.unsqueeze(z_sample, 0)
            im_decoded = model.decode(z_sample)
            im_decoded = im_decoded.view(3, img_size, img_size)
            im_decoded = im_decoded.permute([1, 2, 0])
            #im2 = im_decoded.detach().cpu()
            #im2 *= 255
            #im2 = im2.type(torch.uint8).numpy()
            im_decoded = im_decoded.detach().cpu().numpy()
            #if i == 0 and j == 0:
            #    Image.fromarray(im2).show()
            #if i == len(grid_x)-1 and j == len(grid_y)-1:
            #   Image.fromarray(im2).show()
            figure[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size, :] = im_decoded
    plt.figure(figsize=(15, 15))
    plt.imshow(figure)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, n*img_size, img_size))
    ax.set_yticks(np.arange(0, n*img_size, img_size))
    plt.grid(color='black', linewidth='1.2')
    plt.show()


#todo make wit list of values axes to block
def show_2d_manifold_with_fixed_axis(img_size, latent_size, free_axis_1, free_axis_2,vae_weights_path,no_cuda=False,
                                     seed=1,fixed_prob_val=0.5):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")
    model = VAE_SB(device, img_size=img_size, latent_size=latent_size).to(device)
    checkpoint = torch.load(vae_weights_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    n = 20  # figure with nxn images
    figure = np.zeros((img_size * n, img_size * n, 3))
    # Contruct grid of latent variable values.
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n, endpoint=True))#with probabilities to values of the distribution
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n, endpoint=True))
    #grid_x = np.linspace(-2, 2, n, endpoint=True)#with values of the distribution
    #grid_y = np.linspace(-2, 2, n, endpoint=True)
    # Decode for each square in the grid.
    fixed_val = norm.ppf(fixed_prob_val)
    for i, xi in enumerate(grid_x):
        for j, yj in enumerate(grid_y):
            z_sample = np.repeat(fixed_val, latent_size)
            z_sample[free_axis_1] = xi
            z_sample[free_axis_2] = yj
            z_sample = torch.from_numpy(z_sample).to(device).float()
            z_sample = torch.unsqueeze(z_sample, 0)
            im_decoded = model.decode(z_sample)
            im_decoded = im_decoded.view(3, img_size, img_size)
            im_decoded = im_decoded.permute([1, 2, 0])
            #im2 = im_decoded.detach().cpu()
            #im2 *= 255
            #im2 = im2.type(torch.uint8).numpy()
            im_decoded = im_decoded.detach().cpu().numpy()
            #if i == 0 and j == 0:
            #    Image.fromarray(im2).show()
            #if i == len(grid_x)-1 and j == len(grid_y)-1:
            #   Image.fromarray(im2).show()
            figure[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size, :] = im_decoded
    plt.figure(figsize=(15, 15))
    plt.imshow(figure)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, n*img_size, img_size))
    ax.set_yticks(np.arange(0, n*img_size, img_size))
    plt.grid(color='black', linewidth='1.2')
    plt.show()


#todo make wit list of values axes to block
def show_1d_manifold(img_size, latent_size, vae_weights_path, no_cuda=False, seed=1):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")
    model = VAE_SB(device, img_size=img_size, latent_size=latent_size).to(device)
    checkpoint = torch.load(vae_weights_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    n = 20  # figure with nxn images
    figure = np.zeros((img_size*n,img_size, 3))

    # Contruct grid of latent variable values.
    grid_x = norm.ppf(np.linspace(0.005, 0.995, n, endpoint=True))  # with probabilities to values of the distribution
    # grid_x = np.linspace(-2, 2, n, endpoint=True)#with values of the distribution

    # Decode for each square in the grid.
    for i, xi in enumerate(grid_x):
        z_sample = np.array([xi])
        z_sample = torch.from_numpy(z_sample).to(device).float()
        im_decoded = model.decode(z_sample)
        im_decoded = im_decoded.view(3, img_size, img_size)
        im_decoded = im_decoded.permute([1, 2, 0])
        # im2 = im_decoded.detach().cpu()
        # im2 *= 255
        # im2 = im2.type(torch.uint8).numpy()
        im_decoded = im_decoded.detach().cpu().numpy()
        # if i == 0 and j == 0:
        #    Image.fromarray(im2).show()
        # if i == len(grid_x)-1 and j == len(grid_y)-1:
        #   Image.fromarray(im2).show()
        figure[i * img_size: (i + 1) * img_size, :, :] = im_decoded

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, img_size, img_size))
    ax.set_yticks(np.arange(0, n * img_size, img_size))
    plt.grid(color='black', linewidth='1.2')

    plt.show()

'''if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='gym env id', type=str)

    parser.add_argument('--enc_type', help='the type of attribute that we want to generate/encode', type=str,
                        default='goal', choices=['goal', 'obstacle', 'obstacle_sizes', 'goal_sizes'])
    parser.add_argument('--batch_size', help='numer of batch to train', type=np.float, default=16.)
    parser.add_argument('--train_epochs', help='number of epochs to train vae', type=np.int32, default=20)
    parser.add_argument('--img_size', help='size image in pixels', type=np.int32, default=84)
    parser.add_argument('--latent_size', help='latent size to train the VAE', type=np.int32, default=5)
    parser.add_argument('--beta', help='beta val for the reconstruction loss', type=np.float, default=2.)

    args = parser.parse_args()

    # get names corresponding folders, and files where to store data
    make_dir(this_file_dir+'results/', clear=False)
    base_data_dir = this_file_dir + '../data/'
    data_dir = base_data_dir + args.env + '/'
    train_file = data_dir + train_file_name[args.enc_type]
    weights_path = data_dir + vae_sb_weights_file_name[args.enc_type]
    data_set = np.load(train_file)
    from PIL import Image
    im = Image.fromarray(data_set[0].astype(np.uint8))
    im.show()
    im.close()


    train_Vae(batch_size=args.batch_size, img_size=args.img_size, latent_size=args.latent_size,beta=args.beta,
              train_file=train_file, vae_weights_path=weights_path, epochs=args.train_epochs, load=False)
    show_2d_manifold_with_fixed_axis(img_size=args.img_size,latent_size=args.latent_size, free_axis_1=9, free_axis_2=5,
                                     vae_weights_path=weights_path)
    for v in np.linspace(start=0.05, stop=0.95, num=8):
        show_2d_manifold_with_fixed_axis(img_size=args.img_size,latent_size=args.latent_size, free_axis_1=9, free_axis_2=4,
                                     vae_weights_path=weights_path, fixed_prob_val=v)
    # show_1d_manifold()
    #show_2d_manifold(84)
    print('Successfully trained VAE')'''
