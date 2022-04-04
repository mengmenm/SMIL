class IMDbFuse(nn.Module):
    """docstring forLenet5 Sound"""
    def __init__(self,extractor1, extractor2, extractor_grad=False):
        super(IMDbFuse, self).__init__()
        
        self.image_extractor = extractor1
        self.text_extractor = extractor2

        self.fc1 = nn.Linear(4096, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 23)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()

        if not extractor_grad:
            # for p in self.image_extractor.parameters():
            #     p.requires_grad_(False)

            for p in self.text_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, image_feature, text_feature=None, encoder=None, text_mean=None, noise_layer=['conv1','conv2','fc0','fc1','fc2'], meta_train=True, mode='one'):

        if mode == 'one':
            assert text_mean is not None
            assert noise_layer is not None
            assert encoder is not None

            _, image_feature = self.image_extractor(image_feature, encoder=encoder, noise=True,meta_train=meta_train, noise_layer=['conv1','conv2','fc0','fc1','fc2'])
            text_feature = text_mean.expand(image_feature.shape[0], text_mean.shape[0])# sound feature: 320 dim vector

            image_feature = image_feature.view(image_feature.size(0), -1)  
            text_feature = text_feature.view(text_feature.size(0), -1)
            
            fc1 = {'fc1':image_feature}
            
            x = torch.cat([image_feature, text_feature], dim=1)

            if 'fc1' in noise_layer: # add noise to the concatenated feature: 480 dimension. 
                x = self.softplus(encoder(fc1, meta_train=meta_train)['fc1']) * x
                # print('add noise to fc1')
            f = x

            fc2 = {'fc2':x}
            x = self.fc1(x)
            
            if 'fc2' in noise_layer: # add noise to fc: 64 dimension. 
                x = self.softplus(encoder(fc2, meta_train=meta_train)['fc2']) * x
                # print('add noise to fc2')

            f1 = x
            x = self.relu(x)
            x = self.dropout(x)
            x = self.bn1(x)

            fc3 = {'fc3':x}
            x = self.fc2(x)
            if 'fc3' in noise_layer: # add noise to fc: 64 dimension. 
                x = self.softplus(encoder(fc3, meta_train=meta_train)['fc3']) * x
                # print('add noise to fc3')
            f2 = x
            x = self.relu(x)
            x = self.dropout(x)
            x = self.bn2(x)            

            x = self.fc3(x)

            return x, f, f1, f2

        elif mode == 'two':
            assert text_feature is not None
            _, image_feature = self.image_extractor(image_feature,  noise=False)
            _, text_feature = self.text_extractor(text_feature)

            # p1d = nn.ConstantPad1d(1898, 0)
            image_feature = image_feature.view(image_feature.size(0), -1)  
            text_feature = text_feature.view(text_feature.size(0), -1)
            # text_feature = p1d(text_feature)
            
            x = torch.cat([image_feature, text_feature], dim=1)
            f = x

            x = self.fc1(x)
            f1 = x 
            x = self.relu(x)
            x = self.dropout(x)
            x = self.bn1(x)

            x = self.fc2(x)
            f2 = x
            x = self.relu(x)
            x = self.dropout(x)
            x = self.bn2(x)
            

            x = self.fc3(x)
            return x, f, f1, f2
            # return x, sound_feature
        else:
            raise ValueError('mode should be one or two.')
