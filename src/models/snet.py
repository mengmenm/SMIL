import torch.nn as nn

class SNet(nn.Module):

    def __init__(self, ):
        super(SNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 5, kernel_size=(2, 2))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        

        self.conv2 = nn.Conv2d(5, 10, kernel_size=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        # self.bn3 = nn.BatchNorm2d(64)
        # self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))


        self.fc1 = nn.Linear(160, 10)
        self.fc2 = nn.Linear(10, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu(x)
       
 
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.pool2(x)
        x = self.relu(x)
        f = x
        

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x, f 

class SNetPluse(nn.Module):

    def __init__(self, ):
        super(SNetPluse, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)
        # self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        

        self.conv2 = nn.Conv2d(32, 48, kernel_size=(2, 2))
        self.bn2 = nn.BatchNorm2d(48)
        # self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(48, 20, kernel_size=(2, 2))
        self.bn3 = nn.BatchNorm2d(20)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(4, 4))


        self.fc1 = nn.Linear(320, 160)
        # self.bn5 = nn.BatchNorm1d(160)
        self.fc2 = nn.Linear(160, 32)
        self.fc3 = nn.Linear(32, 10)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
       
 
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.pool3(x)
        x = self.dropout1(x)

        # f = x
        

        x = x.view(x.size(0), -1)
        f = x
        
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        # f = x

        

        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu(x)
        # f = x
        
        x = self.fc3(x)

        return x, f 


if __name__ == '__main__':
    import torch.backends.cudnn as cudnn
    import torch
    from torchsummary import summary

    cudnn.benchmark = True
    cuda = torch.device("cuda")
    cnn = SNetPluse().to(cuda)
    summary(cnn, (1, 20, 20))
    # x = cnn(inn)