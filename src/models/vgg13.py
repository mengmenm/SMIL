import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

class VGG13(nn.Module):

    def __init__(self, output_layers = ['default']):
        super(VGG13, self).__init__()
        self.output_layers = output_layers

        self.conv11 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        

        self.conv21 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv31 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv41 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv51 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))


        self.fc1 = nn.Linear(20480, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 23)

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, encoder=None, noise=False, meta_train=True, noise_layer=['conv1','conv2']):
        if noise:
            assert encoder is not None

            x = self.conv11(x)
            x = self.relu(x)
            x = self.conv12(x)
            x = self.relu(x)
            x = self.pool1(x)

            x = self.conv21(x)
            x = self.relu(x)
            x = self.conv22(x)
            x = self.relu(x)
            x = self.pool2(x)
         
            x = self.conv31(x)
            x = self.relu(x)
            x = self.conv32(x)
            x = self.relu(x)
            x = self.pool3(x)
            
            x = self.conv41(x)
            x = self.relu(x)
            x = self.conv42(x)
            x = self.relu(x)
            x = self.pool4(x)
        
            x = self.conv51(x)
            x = self.relu(x)
            x = self.conv52(x)
            x = self.relu(x)
            x = self.pool5(x)
     
            x = x.view(x.size(0), -1)
            f = x
            

            x = self.fc1(x)
            x = self.relu(x)
            x= self.dropout(x)

            x = self.fc2(x)
            x = self.relu(x)
            x= self.dropout(x)

            x = self.fc3(x)


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
            x = self.conv11(x)
            x = self.relu(x)
            x = self.conv12(x)
            x = self.relu(x)
            x = self.pool1(x)

            x = self.conv21(x)
            x = self.relu(x)
            x = self.conv22(x)
            x = self.relu(x)
            x = self.pool2(x)
         
            x = self.conv31(x)
            x = self.relu(x)
            x = self.conv32(x)
            x = self.relu(x)
            x = self.pool3(x)
            
            x = self.conv41(x)
            x = self.relu(x)
            x = self.conv42(x)
            x = self.relu(x)
            x = self.pool4(x)
        
            x = self.conv51(x)
            x = self.relu(x)
            x = self.conv52(x)
            x = self.relu(x)
            x = self.pool5(x)
     
            x = x.view(x.size(0), -1)
            f = x
            

            x = self.fc1(x)
            x = self.relu(x)
            x= self.dropout(x)

            x = self.fc2(x)
            x = self.relu(x)
            x= self.dropout(x)

            x = self.fc3(x)

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
    cnn = VGG13().to(device)
    summary(cnn, (3, 256, 160))
