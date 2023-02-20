from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from einops.layers.torch import Rearrange

from model import TDiscriminator, TGenerator
import wandb


class Trainer:
    def __init__(self, dataset_name, log):
        self.log = log
        self.dataset_name = dataset_name
        self.device = 'cuda:3'
        self.latent_dim = 256
        self.g = TGenerator().to(self.device)
        self.d = TDiscriminator().to(self.device)
        self.g_optimizer = optim.Adam(self.g.parameters(), lr=0.0001)
        self.d_optimizer = optim.Adam(self.d.parameters(), lr=0.0001)

        self.loss = nn.CrossEntropyLoss().to(self.device)

        self.fixed_noise = Variable(torch.tensor(np.random.normal(0, 1, (64, self.latent_dim)),
                                                 dtype=torch.float32)).to(self.device)
        torch.save(self.fixed_noise, f'collection/pred_{self.dataset_name}_epoch_{0}.pth')

        if self.log:
            wandb.init(project="TransGAN", entity="ndn")
            wandb.config = {}
            print('\n')

    def train(self, train_iter):
        self.g.train()
        self.d.train()
        for epoch in range(10000):
            train_g_loss = 0
            train_d_loss = 0
            for image, label in tqdm(train_iter):
                image = image.to(self.device)

                one_label = Rearrange('b c -> (b c)', c=1)(torch.full(size=(image.shape[0], 1),
                                                                      fill_value=1, dtype=torch.long)).to(self.device)
                zero_label = Rearrange('b c -> (b c)', c=1)(torch.full(size=(image.shape[0], 1),
                                                                       fill_value=0, dtype=torch.long)).to(self.device)

                self.g_optimizer.zero_grad()
                z = Variable(torch.tensor(np.random.normal(0, 1, (image.shape[0], self.latent_dim)),
                                          dtype=torch.float32)).to(self.device)
                gen_img = self.g(z)

                g_loss = self.loss(self.d(gen_img), one_label)
                g_loss.backward()
                self.g_optimizer.step()

                self.d_optimizer.zero_grad()
                d_real_loss = self.loss(self.d(image), one_label)
                d_fake_loss = self.loss(self.d(gen_img.detach()), zero_label)
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.d_optimizer.step()

                train_g_loss += g_loss
                train_d_loss += d_loss
            train_d_loss /= len(train_iter)
            train_g_loss /= len(train_iter)

            print(f'Epoch:{epoch} \t|| G_Loss: {train_g_loss: .4f} \t|| D_Loss: {train_d_loss: .4f}')

            param = {'g': self.g.state_dict(),
                     'd': self.d.state_dict()}
            self.eval(epoch)
            torch.save(param, f'model_{self.dataset_name}.pth')

            if self.log:
                wandb.log({
                    'g_loss': train_g_loss,
                    'd_loss': train_d_loss,
                })

    @torch.no_grad()
    def eval(self, epoch):
        self.g.eval()
        pred = self.g(self.fixed_noise)
        torch.save(pred, f'collection/pred_{self.dataset_name}_epoch_{epoch + 1}.pth')
