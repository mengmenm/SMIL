import os
import shutil 
import torch
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AvgF1(object):
    """docstring for AcgF1"""
    def __init__(self, arg):
        super(AcgF1, self).__init__()
        self.arg = arg

def save_ckpt(state, path, epoch, best_acc = None, train_acc=None):
    filename = 'epoch_'+ str(epoch) + '_ckpt.path.tar'
    ckpt_path = os.path.join(path + '/' + filename)
    torch.save(state, ckpt_path)
    if best_acc is not None:
        bestname = 'epoch_'+ str(epoch) + '_' +'test_accuracy_' + str(best_acc) +'_train_accuracy_' + str(train_acc) +'_best_model.path.tar'
        shutil.copyfile(ckpt_path, os.path.join( path + '/' + bestname))

def save_ckpt_classifier(state, path, iteration, best_acc = None):
    filename = 'iter_'+ str(iteration) + '_ckpt.path.tar'
    ckpt_path = os.path.join(path + '/' + filename)
    torch.save(state, ckpt_path)
    if best_acc is not None:
        bestname = 'iter_'+ str(iteration) + '_' +'test_accuracy_' + str(best_acc) +'_best_model.path.tar'
        shutil.copyfile(ckpt_path, os.path.join( path + '/' + bestname))


def save_ckpt_inferNet(state, path, iteration):
    filename = 'iter_'+ str(iteration) + '_inferNet_ckpt.path.tar'
    ckpt_path = os.path.join(path + '/' + filename)
    torch.save(state, ckpt_path)