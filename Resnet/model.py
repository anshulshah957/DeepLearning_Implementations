'''
ResNet Implementation from:
K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, 2016, pp. 770-778, doi: 10.1109/CVPR.2016.90.
https://arxiv.org/pdf/1512.03385.pdf

Resnet-18 implementation
'''

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils import data
from load_data import get_cifar_10
from torch.nn.functional import cross_entropy
from tensorboardX import SummaryWriter


class Res_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample = False):
        self.downsample = downsample

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        i = x

        x = self.conv(x)

        if self.downsample:
            x = self.downsample_layer(x)
            
        x = x + i
        x = self.relu(x)

        return x

#TODO: Add weight initializations
class ResNet18(nn.Module):
    def __init__(self, image_width=60, image_height=60, image_channels=3, num_classes=10):
        self.input_channels = image_channels
        self.input_height = image_height
        self.input_width - image_width
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.res_layers = nn.Sequential(
            Res_Block(64, 64),
            Res_Block(64, 64),
            Res_Block(64, 128, stride=2, downsample=True),
            Res_Block(128, 128),
            Res_Block(128, 256, stride=2, downsample=True),
            Res_Block(256, 256),
            Res_Block(256, 512, stride=2, downsample=True),
            Res_Block(512, 512),
        )

        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

        self.FC = nn.Linear(512, self.num_classes)
         
    def forward(self, x):
        x = self.conv1(x)
        x = self.res_layers(x)

        x = self.GAP(x)
        x = torch.flatten(x, 1)

        x = self.FC(x)

        return x

    def load_dataset(self, X_train, Y_train, X_test, Y_test, batch_size=128):
        #TODO: add data augmentation
        train_dataset = TensorDataset(X_train, Y_train)
        test_dataset = TensorDataset(X_test, Y_test)

        self.train_dataloader = data.DataLoader(train_dataset, shuffle=True, pin_memory=True, num_workers=8, drop_last=True, batch_size=batch_size)
        self.test_dataloader = data.DataLoader(test_dataset, shuffle=True, pin_memory=True, num_workers=8, drop_last=True, batch_size=batch_size)
    
    #TODO: Update train script to resnet specifications
    def train(self, device, learning_rate=0.01, learning_momentum=0.9, learning_decay=0.0005, num_epochs=90, logdir = os.path.abspath("logs"), checkdir=os.path.abspath("checkpoint")):
        initial_seed = torch.initial_seed()

        
        optimizer = optim.SGD(params=self.parameters(), lr=learning_rate, momentum=learning_momentum, weight_decay=learning_decay)

        tbwriter = SummaryWriter(log_dir=logdir)

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1)
        
        #TODO: divide the learning rate by 10 when the validation error rate stops improving with the current learning rate
        steps = 0
        for epoch in range(1, num_epochs + 1):
            print("Epoch: " + str(epoch))
            lr_scheduler.step()
            for imgs, classes in self.train_dataloader:
                imgs = imgs.to(device)

                classes = classes.tolist()

                for i, one_hot in enumerate(classes):
                    classes[i] = [j for (j, elem) in enumerate(one_hot) if (elem == 1)][0]

                classes = torch.from_numpy(np.array(classes))
                classes = classes.to(device)

                pred = self(imgs)

                loss = cross_entropy(pred, classes)

                optimizer.zero_grad()
                loss.backward()
                
                #Uncomment below to debug gradient update
                #for param in self.parameters():
                    #print(param.grad.data.sum())


                optimizer.step()
                
                #Temporary debugging, add log to something like tensorboard later
                print("Loss:")
                print(loss.item())

                if steps % 8 == 0:
                    with torch.no_grad():
                        _, preds = torch.max(pred, 1)

                        print('Epoch: {} \tStep: {} \tLoss: {:.4f}'.format(epoch, steps, loss.item()))

                        tbwriter.add_scalar('loss', loss.item(), steps)


                steps += 1

            
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(checkdir, 'alexnet_weights_epoch_2_{}.pkl'.format(epoch))

                state = {
                    'epoch': epoch,
                    'total_steps': steps,
                    'optimizer': optimizer.state_dict(),
                    'model': self.state_dict(),
                    'seed': initial_seed,
                }

                torch.save(state, checkpoint_path)


if __name__ == "__main__":
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    resnet = ResNet18(70, 70).to(device)

    X_train, Y_train, X_test, Y_test = get_cifar_10(70, 70)

    resnet.load_dataset(torch.Tensor(X_train), torch.Tensor(Y_train), torch.Tensor(X_test), torch.Tensor(Y_test))
    
    resnet.train(device)

