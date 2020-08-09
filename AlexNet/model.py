'''
AlexNet Implementation from:
"ImageNet Classification with Deep Convolutional Neural Networks" by Krizhevsky et al.
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
'''

#TODO: Add imports

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils import data
from load_data import get_cifar_10
from torch.nn.functional import cross_entropy

class AlexNet(nn.Module):
    #TODO: Calculate correct conv_output_size
    def __init__(self, image_width=60, image_height=60, image_channels=3, num_classes=10):
        super().__init__()

        self.input_width = image_width
        self.input_height = image_height
        self.input_channels = image_channels
        self.num_classes = num_classes

        self.conv = nn.Sequential(
                nn.Conv2d(in_channels=self.input_channels, out_channels=96, kernel_size=11, stride=4),
                nn.ReLU(),
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
        )

        x = np.zeros((3, self.input_width, self.input_height))
        x = torch.Tensor(x)
        x = x.unsqueeze(0)

        conv_output_size = self.conv(x).numel()

        self.conv_output_size = conv_output_size

        self.FC = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=conv_output_size, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

        '''
        Weight and Bias initializations according to paper
        '''

        nn.init.normal_(self.conv[0].weight, mean=0, std=0.01)
        nn.init.normal_(self.conv[4].weight, mean=0, std=0.01)
        nn.init.normal_(self.conv[8].weight, mean=0, std=0.01)
        nn.init.normal_(self.conv[10].weight, mean=0, std=0.01)
        nn.init.normal_(self.conv[12].weight, mean=0, std=0.01)

        nn.init.constant_(self.conv[0].bias, 0)
        nn.init.constant_(self.conv[4].bias, 1)
        nn.init.constant_(self.conv[8].bias, 0)
        nn.init.constant_(self.conv[10].bias, 1)
        nn.init.constant_(self.conv[12].bias, 1)

        nn.init.normal_(self.FC[1].weight, mean=0, std=0.01)
        nn.init.normal_(self.FC[4].weight, mean=0, std=0.01)
        nn.init.normal_(self.FC[6].weight, mean=0, std=0.01)

        nn.init.constant_(self.FC[1].bias, 1)
        nn.init.constant_(self.FC[4].bias, 1)
        nn.init.constant_(self.FC[6].bias, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.conv_output_size)
        x = self.FC(x) 

        return x

    #TODO: add load and checkpointint in training
    def load_weights(self):
        pass

    def load_dataset(self, X_train, Y_train, X_test, Y_test, batch_size=128):
        #TODO: add data augmentation
        train_dataset = TensorDataset(X_train, Y_train)
        test_dataset = TensorDataset(X_test, Y_test)

        self.train_dataloader = data.DataLoader(train_dataset, shuffle=True, pin_memory=True, num_workers=8, drop_last=True, batch_size=batch_size)
        self.test_dataloader = data.DataLoader(test_dataset, shuffle=True, pin_memory=True, num_workers=8, drop_last=True, batch_size=batch_size)

 
    #TODO: Use multiple GPUs
    def train(self, device, learning_rate=0.01, learning_momentum=0.9, learning_decay=0.0005, num_epochs=90):
        
        optimizer = optim.SGD(params=self.parameters(), lr=learning_rate, momentum=learning_momentum, weight_decay=learning_decay)
        
        #TODO: divide the learning rate by 10 when the validation error rate stops improving with the current learning rate
        steps = 0
        for epoch in range(0, num_epochs):
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

if __name__ == "__main__":
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    alexNet = AlexNet(70, 70).to(device)

    X_train, Y_train, X_test, Y_test = get_cifar_10(70, 70)

    alexNet.load_dataset(torch.Tensor(X_train), torch.Tensor(Y_train), torch.Tensor(X_test), torch.Tensor(Y_test))
    
    alexNet.train(device)

