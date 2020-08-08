'''
AlexNet Implementation from:
"ImageNet Classification with Deep Convolutional Neural Networks" by Krizhevsky et al.
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
'''

#TODO: Add imports

import os
import torch
import torch.nn as nn
import torch.optim as optim

class AlexNet(nn.module):
    #TODO: Calculate correct conv_output_size
    def __init__(self, image_width=60, image_height=60, image_channels=3, num_classes=10, conv_output_size=(6 * 6 * 256)):
        super.__init__()

        self.input_width = image_width
        self.input_height = image_height
        self.input_channels = image_channels
        self.num_classes = num_classes
        self.conv_output_size = conv_output_size

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
        Weight initializations according to paper
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

    def load_weights(self):
        pass

    def train(self, learning_rate=0.01, momentum=0.9):
        pass

if __name__ == "__main__":
    pass


