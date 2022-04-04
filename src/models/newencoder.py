import torch.nn as nn
from collections import OrderedDict
import torch

class InferNet(nn.Module):

    def __init__(self, output_layers = ['default']):
        super(InferNet, self).__init__()
        self.output_layers = output_layers

        self.conv1 = nn.Conv2d(1, 5, kernel_size=(5, 5))
        # self.conv11 = nn.Conv2d(1, 5, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(5, 10, kernel_size=(5, 5))
        # self.conv21 = nn.Conv2d(5, 10, kernel_size=(5, 5))

        self.fc1 = nn.Linear(160, 480)
        self.fc11 = nn.Linear(160, 480)

        self.fc2 = nn.Linear(480, 64)
        self.fc21 = nn.Linear(480, 64)


    def forward(self, x, meta_train=True):

        outputs = OrderedDict()
        if meta_train:
            # print('encoder meta_train=True')
            if 'conv1' in x.keys():
                mu1 = self.conv1(x['conv1'])
                # std1 = torch.exp(0.5*mu1)
                # outputs['conv1'] = torch.randn_like(std1)*std1
                outputs['conv1'] = torch.randn_like(mu1) + mu1

            if 'conv2' in x.keys():
                mu2 = self.conv2(x['conv2'])
                # std2 = torch.exp(mu2)
                # outputs['conv2'] = torch.randn_like(std2)*std2
                outputs['conv2'] = torch.randn_like(mu2) + mu2

            if 'fc1' in x.keys():
                mu3 = self.fc1(x['fc1'])
                # sigma3 = self.fc1(x['fc1'])
                # std3 = torch.exp(0.5*sigma3)
                # outputs['fc1'] = torch.randn_like(std3)*std3
                outputs['fc1'] = torch.randn_like(mu3) + mu3
                # outputs['fc1'] =  mu3
                # print(outputs['fc1'])

            if 'fc2' in x.keys():
                mu4 = self.fc2(x['fc2'])
                # sigma4 = self.fc2(x['fc2'])
                # std4 = torch.exp(0.5*sigma4)
                outputs['fc2'] = torch.randn_like(mu4) + mu4
                # outputs['fc2'] =  mu4

            return outputs

        else:
            # print('encoder meta_train=False')
            if 'conv1' in x.keys():
                mu1 = self.conv1(x['conv1'])

                outputs['conv1'] = mu1

            if 'conv2' in x.keys():
                mu2 = self.conv2(x['conv2'])
                outputs['conv2'] = mu2

            if 'fc1' in x.keys():
                mu3 = self.fc1(x['fc1']) 
                outputs['fc1'] = mu3

            if 'fc2' in x.keys():
                mu4 = self.fc2(x['fc2'])
                outputs['fc2'] = mu4

            return outputs

        
class InferNetNew(nn.Module):

    def __init__(self, output_layers = ['default']):
        super(InferNetNew, self).__init__()
        self.output_layers = output_layers

        self.conv1 = nn.Conv2d(1, 5, kernel_size=(5, 5))
        # self.conv11 = nn.Conv2d(1, 5, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(5, 10, kernel_size=(5, 5))
        # self.conv21 = nn.Conv2d(5, 10, kernel_size=(5, 5))

        self.fc0 = nn.Linear(160, 10)

        self.fc1 = nn.Linear(160, 480)
        self.fc11 = nn.Linear(160, 480)

        self.fc2 = nn.Linear(480, 64)
        self.fc21 = nn.Linear(480, 64)




    def forward(self, x, meta_train=True):

        outputs = OrderedDict()
        if meta_train:
            # print('encoder meta_train=True')
            if 'conv1' in x.keys():
                mu1 = self.conv1(x['conv1'])
                # std1 = torch.exp(0.5*mu1)
                
                outputs['conv1'] = torch.randn_like(mu1) + mu1

            if 'conv2' in x.keys():
                mu2 = self.conv2(x['conv2'])
                # std2 = torch.exp(mu2)
                
                outputs['conv2'] = torch.randn_like(mu2) + mu2

            if 'fc0' in x.keys():
                mu0 = self.fc0(x['fc0'])
                outputs['fc0'] = mu0

            if 'fc1' in x.keys():
                mu3 = self.fc1(x['fc1'])
                sigma3 = self.fc11(x['fc1'])
                std3 = torch.exp(0.5*sigma3)
                std3 = torch.clamp(std3, min=0, max=1)
                
                outputs['fc1'] = torch.randn_like(mu3) + mu3

            if 'fc2' in x.keys():
                mu4 = self.fc2(x['fc2'])
                sigma4 = self.fc21(x['fc2'])
                std4 = torch.exp(0.5*sigma4)
                std4 = torch.clamp(std4, min=0, max=1)
                
                outputs['fc2'] = torch.randn_like(mu4) + mu4

            return outputs

        else:
            # print('encoder meta_train=False')
            if 'conv1' in x.keys():
                mu1 = self.conv1(x['conv1'])

                outputs['conv1'] = mu1

            if 'conv2' in x.keys():
                mu2 = self.conv2(x['conv2'])
                outputs['conv2'] = mu2


            if 'fc0' in x.keys():
                mu0 = self.fc0(x['fc0'])
                outputs['fc0'] = mu0

            if 'fc1' in x.keys():
                mu3 = self.fc1(x['fc1']) 
                outputs['fc1'] = mu3

            if 'fc2' in x.keys():
                mu4 = self.fc2(x['fc2'])
                outputs['fc2'] = mu4

            return outputs





if __name__ == '__main__':
    import torch.backends.cudnn as cudnn
    import torch
    from torchsummary import summary

    cudnn.benchmark = True
    device = torch.device("cuda")
    encoder = InferNet().to(device)
    images = torch.rand(8, 1, 28, 28).to(device)
    a = {'conv1':images}
    x = encoder(a,  meta_train=True)
    print(x.keys())
    # summary(encoder, (1, 28, 28))
    # for i,p in enumerate(cnn.parameters()):
    #     if i < 6:
    #         p.requires_grad = False

    for name, param in encoder.named_parameters():
        # if param.requires_grad:
        print(name, param.data.shape)
