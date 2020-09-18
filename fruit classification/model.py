import torch.nn as nn
import torch.nn.functional as F

class FNet(nn.Module):
    """Fruit Net

    """
    def __init__(self, num_classes = 5):
        # make your convolutional neural network here
        # use regularization
        # batch normalization
        super(FNet,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(50*50*8, 300) 
        #self.activ=nn.ReLU()
        self.activ = nn.ReLU()
        self.regu=nn.Dropout(0.1)
        self.fc2=nn.Linear(300,5)
        pass

    def forward(self, x):
        # forward propagation
        out = self.layer1(x)
        # out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out=self.regu(self.activ(out))
        #print(out)
        out=self.fc2(out)
        #print(out)
        return out

if __name__ == "__main__":
    net = FNet()
