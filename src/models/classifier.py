import torch.nn as nn
from collections import OrderedDict
import torch

class ClassfierNet(nn.Module):

    def __init__(self, output_layers = ['default']):
        super(ClassfierNet, self).__init__()
        self.output_layers = output_layers

        self.fc1 = nn.Linear(160, 32)
        self.fc2 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, noise=None):

        x = x.view(x.size(0), -1)
        if noise is None:
            # print('in none')
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            f = x
            x = self.fc2(x)

            return x, f

        else:

            x = self.fc1(x)
            x = self.relu(x)
            if 'fc1' in noise.keys():
                # x = x + noise['fc1']
                x = x * noise['fc1']  
                f = x

            x = self.fc2(x)
            if 'fc2' in noise.keys():
                # x = x + noise['fc2']
                x = x * noise['fc2'] 

            return x, f


class ClassfierNetNew(nn.Module):

    def __init__(self, output_layers = ['default']):
        super(ClassfierNetNew, self).__init__()
        self.output_layers = output_layers

        self.fc1 = nn.Linear(160, 32)
        self.fc2 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        # x = x 
        f = x
        x = self.fc2(x)

        return x, f

if __name__ == '__main__':
    import torch.backends.cudnn as cudnn
    import torch
    from torchsummary import summary

    cudnn.benchmark = True
    device = torch.device("cuda")
    cnn = ClassfierNet().to(device)
    images = torch.rand(8, 160).to(device)
    x = cnn(images)
    summary(cnn, (1, 160))
