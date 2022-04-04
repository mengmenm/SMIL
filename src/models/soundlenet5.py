import torch
import torch.nn as nn

class SoundLenet5(nn.Module):
    """docstring forLenet5 Sound"""
    def __init__(self, extractor1, extractor2, extractor_grad=False):
        super(SoundLenet5, self).__init__()

        self.img_extractor = extractor1
        self.sound_extractor = extractor2

        self.fc1 = nn.Linear(480, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.6)
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()

        if not extractor_grad:


            for p in self.sound_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, label, image, sound=None, encoder=None, sound_mean=None, noise_layer=['conv1','conv2','fc1','fc2'], meta_train=True, mode='one'):

        if mode == 'one':
            assert sound_mean is not None
            assert noise_layer is not None
            assert encoder is not None

            _, img_feature = self.img_extractor(image, encoder=encoder, noise=True, meta_train=meta_train, noise_layer=noise_layer) # image feature: 160 dim vector
            sound_feature = sound_mean.expand(img_feature.shape[0], sound_mean.shape[0])# sound feature: 320 dim vector
            # sound_feature = torch.ones(img_feature.shape[0], sound_mean.shape[0]).to('cuda')
            img_feature = img_feature.view(img_feature.size(0), -1)  
            sound_feature = sound_feature.view(sound_feature.size(0), -1)
            fc1 = {'fc1':img_feature}
            
            x = torch.cat([img_feature, sound_feature], dim=1)
            # fc1 = {'fc1':x}
            if 'fc1' in noise_layer: # add noise to the concatenated feature: 480 dimension. 
                x = self.softplus(encoder(fc1, meta_train=meta_train)['fc1']) + x

            f = x

            fc2 = {'fc2':x}
            x = self.relu(self.fc1(x))
            if 'fc2' in noise_layer: # add noise to fc: 64 dimension. 
                x = self.softplus(encoder(fc2, meta_train=meta_train)['fc2']) + x

            f1 = x
            x = self.dropout(x)
            

            x = self.fc2(x)
            return x, f, f1, sound_feature

        elif mode == 'two':
            assert sound is not None
            _, img_feature = self.img_extractor(image, encoder=None, noise=False, noise_layer=noise_layer)
            _, sound_feature = self.sound_extractor(sound)


            img_feature = img_feature.view(img_feature.size(0), -1)

            sound_feature = sound_feature.view(sound_feature.size(0), -1)
            # print(sound_feature.shape)

            x = torch.cat([img_feature, sound_feature], dim=1)
            f = x
            print(f.shape, x.shape)

            x = self.relu(self.fc1(x))
            f1 = x
            x = self.dropout(x)
            

            x = self.fc2(x)
            return x, f, f1, sound_feature, img_feature
            # return x, sound_feature
        else:
            raise ValueError('mode should be one or two.')
class SoundLenet5AE(nn.Module):
    """docstring forLenet5 Sound"""
    def __init__(self, extractor1, extractor2, extractor_grad=False):
        super(SoundLenet5AE, self).__init__()

        self.img_extractor = extractor1
        self.sound_extractor = extractor2

        self.fc1 = nn.Linear(480, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.6)
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()

        if not extractor_grad:
            # for p in self.img_extractor.parameters():
            #     p.requires_grad_(False)

            for p in self.sound_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, label, image, sound=None, encoder=None, autoencoder=None, noise_layer=['conv1','conv2','fc1','fc2'], meta_train=True, mode='one'):

        if mode == 'one':
            assert autoencoder is not None
            assert noise_layer is not None
            assert encoder is not None

            _, img_feature = self.img_extractor(image, encoder=encoder, noise=True, meta_train=meta_train, noise_layer=noise_layer) # image feature: 160 dim vector
            # sound_feature = sound_mean.expand(img_feature.shape[0], sound_mean.shape[0])# sound feature: 320 dim vector

            autoencoder.eval()
            sound_feature = autoencoder(img_feature).detach()

            # sound_feature = torch.ones(img_feature.shape[0], sound_mean.shape[0]).to('cuda')
            img_feature = img_feature.view(img_feature.size(0), -1)  
            sound_feature = sound_feature.view(sound_feature.size(0), -1)
            fc1 = {'fc1':img_feature}
            
            x = torch.cat([img_feature, sound_feature], dim=1)
            # fc1 = {'fc1':x}
            if 'fc1' in noise_layer: # add noise to the concatenated feature: 480 dimension. 
                x = self.softplus(encoder(fc1, meta_train=meta_train)['fc1']) + x

            f = x

            fc2 = {'fc2':x}
            x = self.relu(self.fc1(x))
            if 'fc2' in noise_layer: # add noise to fc: 64 dimension. 

                x = self.softplus(encoder(fc2, meta_train=meta_train)['fc2']) + x
            f1 = x
            x = self.dropout(x)
            

            x = self.fc2(x)
            return x, f, f1, sound_feature

        elif mode == 'two':
            assert sound is not None
            _, img_feature = self.img_extractor(image, encoder=None, noise=False, noise_layer=noise_layer)
            _, sound_feature = self.sound_extractor(sound)


            img_feature = img_feature.view(img_feature.size(0), -1)

            sound_feature = sound_feature.view(sound_feature.size(0), -1)
            # print(sound_feature.shape)

            x = torch.cat([img_feature, sound_feature], dim=1)
            f = x

            x = self.relu(self.fc1(x))
            f1 = x
            x = self.dropout(x)
            

            x = self.fc2(x)
            return x, f, f1, sound_feature, img_feature
            # return x, sound_feature
        else:
            raise ValueError('mode should be one or two.')


class SoundLenet5New(nn.Module):
    """docstring forLenet5 Sound"""
    def __init__(self, extractor1, extractor2, extractor_grad=False):
        super(SoundLenet5New, self).__init__()

        self.img_extractor = extractor1
        self.sound_extractor = extractor2

        self.fc1 = nn.Linear(480, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

        if not extractor_grad:
            # for p in self.img_extractor.parameters():
            #     p.requires_grad_(False)

            for p in self.sound_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, label, image, sound=None, encoder=None, sound_mean=None, noise_layer=['conv1','conv2','fc1','fc2'], meta_train=True, mode='one'):

        if mode == 'one':
            assert sound_mean is not None
            assert noise_layer is not None
            assert encoder is not None

            _, img_feature = self.img_extractor(image, encoder=encoder, noise=True, meta_train=meta_train, noise_layer=noise_layer) # image feature: 160 dim vector
            # sound_feature = torch.ones(img_feature.shape).to('cuda')
            # print(sound_feature)
            
            sound_mean = sound_mean.expand(img_feature.shape[0], -1, -1)
            
            fc0 = {'fc0':img_feature}
            if 'fc0' in noise_layer: # add noise to the concatenated feature: 480 dimension. 

                
                weight = self.softplus(encoder(fc0, meta_train=meta_train)['fc0']).unsqueeze(-1)

                sound_feature =  sound_mean.matmul(weight)
               
                for i in range(sound_feature.shape[0]):
                    sound_feature[i] = sound_feature[i].clone() / weight.sum(1)[i]





            img_feature = img_feature.view(img_feature.size(0), -1)  
            sound_feature = sound_feature.view(sound_feature.size(0), -1)

            fc1 = {'fc1':img_feature}
            
            x = torch.cat([img_feature, sound_feature], dim=1)

            if 'fc1' in noise_layer: # add noise to the concatenated feature: 480 dimension. 
                
                x = self.softplus(encoder(fc1, meta_train=meta_train)['fc1']) + x
 
            f = x

            fc2 = {'fc2':x}
            x = self.relu(self.fc1(x))
            if 'fc2' in noise_layer: # add noise to fc: 64 dimension. 
                x = self.softplus(encoder(fc2, meta_train=meta_train)['fc2']) + x


            f1 = x
            x = self.dropout(x)
            

            x = self.fc2(x)
            return x, f, f1, sound_feature

        elif mode == 'two':
            assert sound is not None
            _, img_feature = self.img_extractor(image, encoder=None, noise=False, noise_layer=noise_layer)
            _, sound_feature = self.sound_extractor(sound)


            img_feature = img_feature.view(img_feature.size(0), -1)  
            sound_feature = sound_feature.view(sound_feature.size(0), -1)

            x = torch.cat([img_feature, sound_feature], dim=1)
            f = x

            x = self.relu(self.fc1(x))
            f1 = x
            x = self.dropout(x)
            

            x = self.fc2(x)
            return x, f, f1, sound_feature
            # return x, sound_feature
        else:
            raise ValueError('mode should be one or two.')

class SoundLenet5Class(nn.Module):
    """docstring forLenet5 Sound"""
    def __init__(self, extractor1, extractor2, extractor_grad=False):
        super(SoundLenet5Class, self).__init__()

        self.img_extractor = extractor1
        self.sound_extractor = extractor2

        self.fc1 = nn.Linear(480, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

        if not extractor_grad:
            for p in self.sound_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, label, image, sound=None, encoder=None, sound_mean=None, noise_layer=['conv1','conv2','fc1','fc2'], meta_train=True, mode='one'):

        if mode == 'one':
            assert sound_mean is not None
            assert noise_layer is not None
            assert encoder is not None

            _, img_feature = self.img_extractor(image, encoder=encoder, noise=True, meta_train=meta_train, noise_layer=noise_layer) # image feature: 160 dim vector
            sound_feature = torch.ones(img_feature.shape[0], 320).to('cuda')


            for i in range(img_feature.shape[0]):
                sound_feature[i] = sound_mean[label[i].item()]
            



            img_feature = img_feature.view(img_feature.size(0), -1)  
            sound_feature = sound_feature.view(sound_feature.size(0), -1)

            fc1 = {'fc1':img_feature}
            
            x = torch.cat([img_feature, sound_feature], dim=1)
 
            if 'fc1' in noise_layer: # add noise to the concatenated feature: 480 dimension. 
                
                x = self.softplus(encoder(fc1, meta_train=meta_train)['fc1']) + x
                
                # print('add noise to fc1')
            f = x

            fc2 = {'fc2':x}
            x = self.relu(self.fc1(x))
            if 'fc2' in noise_layer: # add noise to fc: 64 dimension. 
                
                x = self.softplus(encoder(fc2, meta_train=meta_train)['fc2']) + x
                
                # print('add noise to fc2')

            f1 = x
            x = self.dropout(x)
            

            x = self.fc2(x)
            return x, f, f1, sound_feature

        elif mode == 'two':
            assert sound is not None
            _, img_feature = self.img_extractor(image, encoder=None, noise=False, noise_layer=noise_layer)
            _, sound_feature = self.sound_extractor(sound)


            img_feature = img_feature.view(img_feature.size(0), -1)  
            sound_feature = sound_feature.view(sound_feature.size(0), -1)

            x = torch.cat([img_feature, sound_feature], dim=1)
            f = x

            x = self.relu(self.fc1(x))
            f1 = x
            x = self.dropout(x)
            

            x = self.fc2(x)
            return x, f, f1, sound_feature
            # return x, sound_feature
        else:
            raise ValueError('mode should be one or two.')

class SoundLenet5mean(nn.Module):
    """docstring forLenet5 Sound"""
    def __init__(self, extractor1, extractor2, extractor_grad=False):
        super(SoundLenet5mean, self).__init__()

        self.img_extractor = extractor1
        self.sound_extractor = extractor2

        self.fc1 = nn.Linear(480, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.6)
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()

        if not extractor_grad:
            # for p in self.img_extractor.parameters():
            #     p.requires_grad_(False)

            for p in self.sound_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, image, sound=None, encoder=None, sound_mean=None, noise_layer=['conv1','conv2','fc1','fc2'], meta_train=True, mode='one'):

        if mode == 'one':
            assert sound_mean is not None
            assert noise_layer is not None
            assert encoder is not None

            _, img_feature = self.img_extractor(image, encoder=encoder, noise=True, meta_train=meta_train, noise_layer=noise_layer) # image feature: 160 dim vector
            sound_feature = sound_mean.expand(img_feature.shape[0], sound_mean.shape[0])# sound feature: 320 dim vector

            img_feature = img_feature.view(img_feature.size(0), -1)  
            sound_feature = sound_feature.view(sound_feature.size(0), -1)
            fc1 = {'fc1':img_feature}
            
            x = torch.cat([img_feature, sound_feature], dim=1)
            # fc1 = {'fc1':x}
            if 'fc1' in noise_layer: # add noise to the concatenated feature: 480 dimension. 
                
                x = self.softplus(encoder(fc1, meta_train=meta_train)['fc1']) + x
                
                # print('add noise to fc1')
            f = x

            fc2 = {'fc2':x}
            x = self.relu(self.fc1(x))
            if 'fc2' in noise_layer: # add noise to fc: 64 dimension. 
                
                x = self.softplus(encoder(fc2, meta_train=meta_train)['fc2']) + x
                
                # print('add noise to fc2')

            f1 = x
            x = self.dropout(x)
            

            x = self.fc2(x)
            return x, f, f1

        elif mode == 'two':
            assert sound is not None
            _, img_feature = self.img_extractor(image, encoder=None, noise=False, noise_layer=noise_layer)
            _, sound_feature = self.sound_extractor(sound)


            img_feature = img_feature.view(img_feature.size(0), -1)  
            sound_feature = sound_feature.view(sound_feature.size(0), -1)

            x = torch.cat([img_feature, sound_feature], dim=1)
            f = x
            # print(f.shape)

            x = self.relu(self.fc1(x))
            f1 = x
            x = self.dropout(x)
            

            x = self.fc2(x)
            # return x, f, f1
            return x, sound_feature
        else:
            raise ValueError('mode should be one or two.')

if __name__ == '__main__':
    import torch.backends.cudnn as cudnn
    import torch
    import sys
    import os.path as path
    sys.path.append('../')
    import numpy as np
    from torchsummary import summary
    from models.lenet5 import LeNet5
    from models.snet import SNetPluse
    # from models.soundlenet5 import SoundLenet5
    from models.newencoder import InferNetNew
    cudnn.benchmark = True
    device = torch.device("cuda")


    e1 = LeNet5().to(device)
    e2 = SNetPluse().to(device)
    
    soundmnist_model_path = '/home/mengma/Desktop/nips2020/src/save/soundmnist/15/2020-04-09T17:05:35.475086/'

    image_sound_extractor = SoundLenet5Class(e1, e2).to(device)

    ckpt_image_sound = torch.load(path.join(soundmnist_model_path, 'epoch_16_test_accuracy_90.0_train_accuracy_100.0_best_model.path.tar'))
    # image_sound_extractor.load_state_dict(ckpt_image_sound['state_dict'])

    # cnn = SoundLenet5(e1, e2).to(device)
    encoder = InferNetNew().to(device)
    path = "/home/mengma/Desktop/nips2020/src/save/sound_mean/kmean/15/epoch_16_test_accuracy_90.0_train_accuracy_100.0_best_model.path.tar/sound_mean_150.npy"
    sound_mean = np.load(path)
    sound_mean = torch.from_numpy(sound_mean).to(device)

    images = torch.rand(5, 1, 28,28).to(device)
    # noise = encoder(images, output_layers=['conv1','conv2','fc1','fc2'], meta_train=True)
    # # print(images[1].size())
    sound = torch.rand(5, 1, 20,20).to(device)
    label = torch.randint(0, 10, (32,)).to(device)
    

    x,f,_,_ = image_sound_extractor(label, images, sound=sound, encoder=encoder,  sound_mean=sound_mean, noise_layer=['fc0','fc1','fc2'], meta_train=False, mode='one')
    # print(x.shape)

    params = list(image_sound_extractor.parameters())
    print(len(params))
    # for i,p in enumerate(cnn.parameters()):
    #     if i < 6:
    #         p.requires_grad = False
    a = 0
    for name, param in image_sound_extractor.named_parameters():
        # if param.requires_grad:
        print(a, name, param.data.shape)
        a += 1