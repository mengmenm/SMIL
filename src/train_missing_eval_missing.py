from __future__ import print_function, absolute_import, division

import os
import time
import math
import datetime
import argparse
import os.path as path
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from models.lenet5 import LeNet5
from models.soundlenet5 import SoundLenet5, SoundLenet5New, SoundLenet5Class
from models.snet import SNetPluse
from models.classifier import ClassfierNet
from models.newencoder import InferNet, InferNetNew
from models.loss import KDFeatureLoss, KDFeatureLossTwo, KDLossAlignTwo

from dataset.meta_training_dataset import MetaTrSouMNIST
from dataset.meta_testing_dataset import MetaTeSouMNIST
from dataset.soundmnist import SoundMNIST
from dataset.mnist import MNIST

from utils.misc import AverageMeter, AvgF1, save_ckpt, save_ckpt_inferNet, save_ckpt_classifier

from metann import ProtoModule

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')
    parser.add_argument('-i', '--image_root', default = '../data/mnist/', type = str, help='data root' )
    parser.add_argument('-s', '--sound_root', default = '../data/sound_450/', type = str, help='data root' )
    parser.add_argument('--checkpoint', default='./save/metadrop_new/feature/15', type=str, help='checkpoint directory')
    parser.add_argument('--snapshot', default=1, type=int, help='save models for every # epochs (default: 1)')

    parser.add_argument('--soundmnist_model_path', default='./save/soundmnist/path/', 
                        type=str, help='pre-trained sound mnist model')
    parser.add_argument('--soundmnist_model_name', default='name/of/best_model.path.tar', 
                        type=str, help='pre-trained sound mnist model')


    parser.add_argument('--sound_mean_path', default='./save/sound_mean/kmean/path/', 
                        type=str, help='pre calculated sound mean path')
    parser.add_argument('--sound_mean_name', default='sound_mean_150.npy', 
                        type=str, help='pre calculated sound mean name')
    

    # model related parm
    parser.add_argument('-b', '--batch_size', default = 128, type = int, help='batch size' )
    parser.add_argument('--per_class_num', default = 15 , type = int, help='per_class_num' ) # 15 * 10 = 150 sound data available, total 1500 sound data
    parser.add_argument('--iterations', default = 8000 , type = int, help='num of epoch' )
    parser.add_argument('--lr', default = 1e-3, type = float, help='initial learning rate' )
    parser.add_argument('--lr_inner', default = 1e-3, type = float, help='initial learning rate' )
    parser.add_argument('--inner_loop', default = 1, type = int, help='meta_train inner_loop' )
    parser.add_argument('--mc_size', default = 30, type = int, help='MC size for meta-test' )
    parser.add_argument('--vis_device', default='0', type=str, help='set visiable device')

    args = parser.parse_args()

    return args

def main(args):
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.vis_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # meta-train step dataset     
    meta_train_dataset = MetaTrSouMNIST(img_root=args.image_root, sound_root=args.sound_root, 
                                        per_class_num=args.per_class_num, meta_split='mtr')
    meta_val_dataset = MetaTrSouMNIST(img_root=args.image_root, sound_root=args.sound_root, 
                                      per_class_num=args.per_class_num, meta_split='mval')


    meta_train_loader = DataLoader(meta_train_dataset, batch_size = args.batch_size, shuffle = True,  
                                   num_workers=0, pin_memory=True)

    meta_val_loader = DataLoader(meta_val_dataset, batch_size = args.batch_size, shuffle = True, 
                                 num_workers=0, pin_memory=True)
    
    # meta-training dataset
    meta_test_dataset = SoundMNIST(img_root=args.image_root,sound_root=args.sound_root, per_class_num=args.per_class_num, train=False)
    meta_test_loader = DataLoader(meta_test_dataset, batch_size = args.batch_size, shuffle = False, num_workers=0, pin_memory=True)
    print('train data size:', len(meta_train_dataset))
    print('val data size:', len(meta_val_dataset))
    print('test data size:',len(meta_test_dataset))


    #creat model 
    print('==> model creating....')

    encoder = InferNetNew().to(device) # auxilary model to infer the missing modality
    encoder = ProtoModule(encoder)

    i_extractor = LeNet5().to(device) # image extractor
    s_extractor = SNetPluse().to(device) # sound extractor

    image_sound_extractor = SoundLenet5New(i_extractor, s_extractor, extractor_grad=False).to(device) # multimodal fusion model

    ckpt_image_sound = torch.load(path.join(args.soundmnist_model_path, args.soundmnist_model_name)) # load pre-trained weight
    image_sound_extractor.load_state_dict(ckpt_image_sound['state_dict'])

    image_sound_extractor = ProtoModule(image_sound_extractor)

    # load pre calculated sound mean
    sound_mean = np.load(os.path.join(args.sound_mean_path, args.sound_mean_name))
    sound_mean = torch.from_numpy(sound_mean).T.to(device)

        
    print('==> mdoel has been created')
    print("==> Total parameters (reference): {:.2f}M".format(sum(p.numel() for p in image_sound_extractor.parameters()) / 1000000.0))
    
    
    criterion_meta_train = nn.CrossEntropyLoss().to(device)
    criterion_meta_val = KDLossAlignTwo(alpha = 0.01, beta = 0.01).to(device)

    optimizer_image_sound = torch.optim.Adam(image_sound_extractor.parameters(), lr = args.lr, weight_decay = 1e-4)
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr = args.lr, weight_decay = 1e-4)

    
    scheduler_image_sound = torch.optim.lr_scheduler.StepLR(optimizer_image_sound, step_size = 5000, gamma = 0.1)  
    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size = 5000, gamma = 0.1)



    best_val_acc =None
    
    glob_step = 0

    evl_step = 0

    ckpt_dir_path = path.join(args.checkpoint, datetime.datetime.now().isoformat())
    # print(ckpt_dir_path)
    if not path.exists(ckpt_dir_path):
            os.makedirs(ckpt_dir_path)
            print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))

    writer = SummaryWriter(log_dir=ckpt_dir_path)

    print('start training')
    for iterate in range(args.iterations):
        # Train for one iteration
        if iterate == 0:
            test_image_acc_i = eval_image_test(meta_test_loader, image_sound_extractor, encoder, sound_mean, device, writer)
            
            mtr_image_acc_i = eval_image_train(meta_train_loader, image_sound_extractor, encoder, sound_mean, device, writer)
            mval_image_acc_i = eval_image_test(meta_val_loader, image_sound_extractor, encoder, sound_mean, device, writer)
            torch.set_grad_enabled(True)
            print('Initial Model  ACC: mtr image acc:{:.4f}, mval image acc:{:.4f}, image-only test acc:{:.4f}' .format(mtr_image_acc_i, mval_image_acc_i, test_image_acc_i))


        meta_train_batch = next(iter(meta_train_loader))

        meta_val_batch = next(iter(meta_val_loader))
        
        glob_step = meta_training(args, meta_train_batch, meta_val_batch, sound_mean, image_sound_extractor, encoder, criterion_meta_train, criterion_meta_val, optimizer_image_sound, optimizer_encoder, device, writer, iterate, args.iterations)

        scheduler_image_sound.step()
        scheduler_encoder.step()

        if (iterate) % 200 == 0:
            
            test_image_acc = 0.
            for i in range(30):
                test_image_acc += eval_image_test(meta_test_loader, image_sound_extractor, encoder, sound_mean, device, writer)
            test_image_acc = test_image_acc / 30

            mtr_image_acc = eval_image_train(meta_train_loader, image_sound_extractor, encoder, sound_mean, device, writer)
            mval_image_acc = eval_image_test(meta_val_loader, image_sound_extractor, encoder, sound_mean, device, writer)
            torch.set_grad_enabled(True)
            print('Iteration:[{}/{}], mtr image acc:{:.6f}, mval image acc:{:.6f}, image-only test acc:{:.6f}' .format(iterate, args.iterations, mtr_image_acc, mval_image_acc, test_image_acc))

        
        # save best model
        if (iterate ) >= 1200:
            
            if best_val_acc is None or best_val_acc < test_image_acc:
                best_val_acc = test_image_acc

                torch.set_grad_enabled(True)
                print('Iteration:[{}/{}], Image-only Test Acc:{:.6f} ' .format(iterate, args.iterations, best_val_acc))
                
                save_ckpt_classifier({'iterate': iterate, 'lr': args.lr, 'state_dict': image_sound_extractor.state_dict(),
                         'optimizer': optimizer_image_sound.state_dict() }, ckpt_dir_path, iteration =  iterate , best_acc = best_val_acc)
                save_ckpt_inferNet({'iterate': iterate, 'lr': args.lr, 'step': glob_step, 'state_dict': encoder.state_dict(), 
                             'optimizer': optimizer_encoder.state_dict() }, ckpt_dir_path, iteration = iterate )
    writer.close()
    

def meta_training(args, meta_train_batch, meta_val_batch, sound_mean, image_sound_extractor, encoder, criterion_meta_train, criterion_meta_val, optimizer_image_sound, optimizer_encoder, device, writer, iterate, total_iterate):
    ''' train one epoch'''

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses= AverageMeter()

    torch.set_grad_enabled(True)

    batch_size = meta_train_batch[0].shape[0]

    # batch of meta train: sampled from image modality
    meta_train_image = meta_train_batch[0].to(device)
    meta_train_label = meta_train_batch[1].to(device)

    # batch of meta validaiton: sampled from both image and sound modality
    meta_val_image = meta_val_batch[0].to(device)
    meta_val_sound = meta_val_batch[1].to(device)
    meta_val_label = meta_val_batch[2].to(device)

    # meta-training 
    params = list(image_sound_extractor.parameters())
    for i in params:
        if not i.requires_grad:
            i.requires_grad = True


    loss_meta_train = 0.
    loss_meta_val = 0.
    mse_loss = nn.MSELoss(reduction='mean')

    for idx in range(args.inner_loop):
        if idx == 0:
            params_star = params

        pred_meta_train_noised,_,_,_ = image_sound_extractor.functional(params_star, True)(meta_train_label, meta_train_image, sound=None, encoder=encoder,  sound_mean=sound_mean, noise_layer=['fc0','fc1','fc2'], meta_train=True, mode='one')

        loss_meta_train = criterion_meta_train(pred_meta_train_noised, meta_train_label)
        torch.autograd.set_detect_anomaly(True)
        grads = torch.autograd.grad(loss_meta_train, params_star, allow_unused=True, create_graph=True)  # create_graph=True: allow second order derivative

        for i in range(len(params_star)):
            if i <= 7 or i >=26:# not update the sound branch
                if grads[i] is not None: # unused parameters have no gradient 
                    params_star[i] = (params_star[i] - args.lr_inner*(0.1**(iterate//1000))*grads[i]).requires_grad_()


    pred_meta_val_noised, f_meta_val_noised1,f_meta_val_noised2, sound_mean_val_noised = image_sound_extractor.functional(params_star, True)(meta_val_label, meta_val_image, sound=None, encoder=encoder, sound_mean=sound_mean, noise_layer=['fc0','fc1','fc2'], meta_train=False, mode='one')

    pred_meta_val_clean, f_meta_val_clean1,f_meta_val_clean2, sound_mean_val_clean = image_sound_extractor.functional(params_star, True)(meta_val_label, meta_val_image, sound=meta_val_sound, mode='two')
    

    sound_mean_val_mse = mse_loss(sound_mean_val_clean, sound_mean_val_noised)

    loss_meta_val = criterion_meta_val(sound_mean_val_clean, sound_mean_val_noised, f_meta_val_clean1, f_meta_val_noised1, f_meta_val_clean2, f_meta_val_noised2, pred_meta_val_noised, pred_meta_val_clean, meta_val_label)

    optimizer_encoder.zero_grad()
    optimizer_image_sound.zero_grad()
    (loss_meta_train + loss_meta_val).backward(create_graph=True)
    optimizer_encoder.step()
    optimizer_image_sound.step()
    torch.cuda.empty_cache()

    if (iterate) % 100 == 0:
        print('Iteration [{}/{}], meta-train Loss: {:.4f}, meta-val Loss: {:.4f},' .format(iterate, total_iterate, 
                                                                                       loss_meta_train.item(), loss_meta_val.item()))

    writer.add_scalar('meta-train loss', loss_meta_train.item(), iterate)
    writer.add_scalar('meta-val loss', loss_meta_val.item(), iterate)
    writer.add_scalar('sound_mean val mse Loss', sound_mean_val_mse.item(), iterate)

    return iterate


def eval_image_test(test_loader, image_sound_extractor, encoder,sound_mean, device, writer):
    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    image_sound_extractor.eval()
    correct = 0
    total = 0
    for i, batch in enumerate(test_loader):

        images = batch[0].to(device)
        sounds = batch[1].to(device)
        labels = batch[2].to(device)

        outputs,_,_,_ = image_sound_extractor(labels, images, sound=None, encoder=encoder,  sound_mean=sound_mean, noise_layer=['fc0','fc1','fc2'], meta_train=False, mode='one')


        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    
    
    return test_acc

def eval_image_train(test_loader, image_sound_extractor, encoder,sound_mean, device, writer):
    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    image_sound_extractor.eval()
    correct = 0
    total = 0
    for i, batch in enumerate(test_loader):

        images = batch[0].to(device)
        labels = batch[1].to(device)

        outputs,_,_,_ = image_sound_extractor(labels, images, sound=None, encoder=encoder,  sound_mean=sound_mean, noise_layer=['fc0','fc1','fc2'], meta_train=False, mode='one')

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    
    
    return test_acc





if __name__ == '__main__':
    main(parse_args())