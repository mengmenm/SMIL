import torch.nn as nn
from collections import OrderedDict
import torch

class InferNetNew(nn.Module):

    def __init__(self, output_layers = ['default']):
        super(InferNetNew, self).__init__()
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

    def _add_meta_train_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = self.softplus(torch.randn_like(x) + x)
        return len(output_layers) == len(outputs)

    def _add_meta_val_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = (x)
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None, meta_train=True):
        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers

        x = self.conv1(x)
        if meta_train:
            if self._add_meta_train_output_and_check('conv1', x, outputs, output_layers):
                return outputs
        else:
            if self._add_meta_val_output_and_check('conv1', x, outputs, output_layers):
                return outputs

        x = self.pool1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.dropout(x)
        if meta_train:
            if self._add_meta_train_output_and_check('conv2', x, outputs, output_layers):
                return outputs
        else:
            if self._add_meta_val_output_and_check('conv2', x, outputs, output_layers):
                return outputs

        x = self.pool2(x)
        x = self.relu(x)
        
        
        

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        if meta_train:
            if self._add_meta_train_output_and_check('fc1', x, outputs, output_layers):
                return outputs
        else:
            if self._add_meta_val_output_and_check('fc1', x, outputs, output_layers):
                return outputs

        x = self.fc2(x)
        if meta_train:
            if self._add_meta_train_output_and_check('fc2', x, outputs, output_layers):
                return outputs
        else:
            if self._add_meta_val_output_and_check('fc2', x, outputs, output_layers):
                return outputs

        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x 

        raise ValueError('output_layer is wrong.')

class InferNet(nn.Module):

    def __init__(self, output_layers = ['default']):
        super(InferNet, self).__init__()
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

    def _add_meta_train_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = self.softplus(torch.randn_like(x) + x)
        return len(output_layers) == len(outputs)

    def _add_meta_val_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = (x)
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None, meta_train=True):
        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers
               

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        if meta_train:
            if self._add_meta_train_output_and_check('fc1', x, outputs, output_layers):
                return outputs
        else:
            if self._add_meta_val_output_and_check('fc1', x, outputs, output_layers):
                return outputs

        x = self.fc2(x)
        if meta_train:
            if self._add_meta_train_output_and_check('fc2', x, outputs, output_layers):
                return outputs
        else:
            if self._add_meta_val_output_and_check('fc2', x, outputs, output_layers):
                return outputs

        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x 

        raise ValueError('output_layer is wrong.')


if __name__ == '__main__':
    import torch.backends.cudnn as cudnn
    import torch
    from torchsummary import summary

    cudnn.benchmark = True
    device = torch.device("cuda")
    encoder = InferNet().to(device)
    images = torch.rand(8, 160).to(device)
    x = encoder(images, output_layers=['fc1'], meta_train=True)
    print(x['fc1'].shape)