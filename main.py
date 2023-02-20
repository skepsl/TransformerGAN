from trainer import Trainer
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets


class Main:
    def __init__(self, dataset_name, log):
        self.dataset = dataset_name
        self.train = Trainer(dataset_name=self.dataset, log=log)

        if dataset_name == 'cifar10':
            self.cifar10()

        else:
            self.celeba()

    def cifar10(self):
        transform_train = transforms.Compose([
            transforms.Resize(32, ),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])

        trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
        testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_train)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
        self.train.train(train_loader)

    def celeba(self):
        transform = transforms.Compose([
                                       transforms.Resize(32,),
                                       transforms.CenterCrop(32,),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])
        dataset = datasets.ImageFolder(root='./data/CelebA', transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
        self.train.train(dataloader)


if __name__ == '__main__':
    Main(dataset_name='cifar10', log=True)
    pass
