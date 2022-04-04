import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

class LeNet5(nn.Module):

    def __init__(self, output_layers = ['default']):
        super(LeNet5, self).__init__()
        self.output_layers = output_layers

        self.conv1 = nn.Conv2d(1, 5, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        

        self.conv2 = nn.Conv2d(5, 10, kernel_size=(5, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))


        self.fc1 = nn.Linear(160, 32)
        self.fc2 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, encoder=None, noise=True, meta_train=True, noise_layer=['conv1','conv2']):
        if noise:
            assert encoder is not None
            conv1 = {'conv1':x}
            x = self.conv1(x)
            if 'conv1' in noise_layer:
                x = self.softplus(encoder(conv1, meta_train=meta_train)['conv1']) + x

            x = self.pool1(x)
            x = self.relu(x)
             
            conv2 = {'conv2':x}
            x = self.conv2(x)
            if 'conv2' in noise_layer:
                x = self.softplus(encoder(conv2, meta_train=meta_train)['conv2']) + x
                
            x = self.dropout(x)
            x = self.pool2(x)
            x = self.relu(x)

            x = x.view(x.size(0), -1)
            f = x
            
            x = self.fc1(x)
            x = self.relu(x)

            x = self.fc2(x)
            # x = x

            return x, f

        else:
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.relu(x)
     
            x = self.conv2(x)
            x = self.dropout(x)
            x = self.pool2(x)
            x = self.relu(x)
            # f = x

            x = x.view(x.size(0), -1)
            f = x
            

            x = self.fc1(x)
            x = self.relu(x)
            # f = x

            x = self.fc2(x)

            return x, f

        




if __name__ == '__main__':
    import torch.backends.cudnn as cudnn
    import torch
    from torchsummary import summary
    import os.path as path
    import sys
    sys.path.append('../')
    from models.newencoder import InferNet

    cudnn.benchmark = True
    device = torch.device("cuda")
    cnn = LeNet5().to(device)
    images = torch.rand(8, 1, 28,28).to(device)
    encoder = InferNet().to(device)
    x,f = cnn(images, encoder, noise=True)
    # print(x.shape)
    summary(cnn, (1, 28, 28))