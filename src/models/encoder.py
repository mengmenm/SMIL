import torch.nn as nn
import torch
from torch.distributions.normal import Normal

class InferenceNet(nn.Module):

    def __init__(self, ):
        super(InferenceNet, self).__init__()

        self.fc1 = nn.Linear(320, 128)
        self.fc2 = nn.Linear(128, 32*2+10*2)
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()

    def forward(self, x, meta_train=True):
        mean, var = torch.mean(x, 0), torch.var(x, 0) 
        # print(mean.shape, var.shape)
        x = torch.cat([mean, var], dim=0)
        # print(x.shape)

        x = self.fc1(x)
        x = self.relu(x)

        mu = self.softplus(self.fc2(x))
        # print(mu.shape)
        if meta_train:            
            new_mu = self.softplus(torch.randn_like(mu) + mu)
            # mu_w = new_mu[:32+10]
            # mu_b = new_mu[32+10:]
        else:
            new_mu = self.softplus(mu)
            # mu_w = new_mu[:32+10]
            # mu_b = new_mu[32+10:]

        return new_mu


if __name__ == '__main__':
    import torch.backends.cudnn as cudnn
    import torch
    import sys
    sys.path.append('../')
    from models.lenet5 import LeNet5

    cudnn.benchmark = True
    cuda = torch.device("cuda")
    cnn = LeNet5().to(cuda)
    for i,p in enumerate(cnn.parameters()):
        if i < 6:
            p.requires_grad = False

    encoder = InferenceNet().to(cuda)

    images = torch.rand(32, 1, 28, 28).to(cuda)
    # print(images[0].shape, images[0].unsqueeze(0).shape)
    x, f = cnn(images)
    # print(f.shape)

    # summary(cnn, (1, 28, 28))
    mu = encoder(f, meta_train=False )


    mu = torch.split(mu, [32,32,10,10])
    print(mu[0])
    print(mu[0][1])

    # for i in mu:
    #     print(i.shape[0])
    # print(mu[2].shape)
    params = list(encoder.parameters())
