import torch.nn as nn
import torch
import torch.nn.functional as F

class KDFeatureLoss(nn.Module):
    """ multi-label cross entropy loss """
    def __init__(self, reduction = 'mean', alpha = 1, beta = 1 ):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction = reduction)
        self.l2_loss = nn.MSELoss(reduction=reduction)
        # self.factor = factor
        self.alpha = alpha
        self.beta = beta


    def forward(self, map_teacher, map_student, pred_student, label):
        loss_ce = self.cross_entropy(pred_student, label)
        loss_dist = self.l2_loss(map_teacher, map_student)
        loss = self.alpha*loss_ce + self.beta*loss_dist

        return loss

class KDFeatureLossTwo(nn.Module):
    """ multi-label cross entropy loss """
    def __init__(self, reduction = 'mean', alpha = 0.1):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction = reduction)
        # self.cross_entropy_clean = nn.CrossEntropyLoss(reduction = reduction)
        self.l2_loss = nn.MSELoss(reduction=reduction)
        # self.factor = factor
        self.alpha = alpha
        # self.beta = beta


    def forward(self, map_teacher1, map_student1, pred_noise, pred_clean, label):
        loss_ce_noise = self.cross_entropy(pred_noise, label)
        loss_ce_clean = self.cross_entropy(pred_clean, label)
        loss_map = self.l2_loss(map_teacher1, map_student1)
        loss = loss_ce_noise + loss_ce_clean + self.alpha*loss_map

        return loss

class KDLossAlignTwo(nn.Module):
    """ multi-label cross entropy loss """
    def __init__(self, reduction = 'mean', alpha = 0.1, beta=0.1):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction = reduction)
        self.l2_loss = nn.MSELoss(reduction=reduction)
        self.alpha = alpha
        self.beta = beta


    def forward(self, mean_teacher, mean_student, map_teacher1, map_student1, map_teacher2, map_student2, pred_noise, pred_clean, label):
        loss_ce_noise = self.cross_entropy(pred_noise, label)
        loss_ce_clean = self.cross_entropy(pred_clean, label)
        loss_map_1 = self.l2_loss(map_teacher1, map_student1)
        loss_map_2 = self.l2_loss(map_teacher2, map_student2)
        loss_audio_mean = self.l2_loss(mean_teacher, mean_student)
        loss = loss_ce_noise  + self.alpha*loss_map_1 + self.beta*loss_map_2 
        

        return loss

class KDPredLoss(nn.Module):
    """ multi-label cross entropy loss """
    def __init__(self, reduction = 'mean', T = 2, alpha=0.5):
        """ pos_weight =  # of neg_sample/ # of pos_sample; it is a tensor vector length equals to num of clss"""
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction = reduction)
        self.kl_loss = nn.KLDivLoss()
        self.T = T
        self.alpha = alpha

    def forward(self, pred_teacher, pred_student, label):
        
        # if weights is not None:
        loss_ce = self.cross_entropy(pred_student, label)
        output_S = F.log_softmax(pred_student/self.T, dim=1)
        output_T = F.softmax(pred_teacher/self.T, dim=1)
        loss_dist = self.kl_loss(output_S, output_T)*self.T*self.T
        loss = (1- self.alpha)*loss_ce + loss_dist*self.alpha
        return loss

class KDLossAll(nn.Module):
    """ multi-label cross entropy loss """
    def __init__(self, reduction = 'mean', T = 2, alpha=0.5, beta = 0.5):
        """ pos_weight =  # of neg_sample/ # of pos_sample; it is a tensor vector length equals to num of clss"""
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction = reduction)
        self.l2_loss = nn.MSELoss(reduction=reduction)
        self.kl_loss = nn.KLDivLoss()
        self.T = T
        self.alpha = alpha
        self.beta = beta

    def forward(self, map_teacher, map_student, pred_teacher,  pred_student, label):
        
        # if weights is not None:
        loss_ce = self.cross_entropy(pred_student, label)
        output_S = F.log_softmax(pred_student/self.T, dim=1)
        output_T = F.softmax(pred_teacher/self.T, dim=1)
        loss_kl = self.kl_loss(output_S, output_T)*self.T*self.T
        loss_l2 = self.l2_loss(map_teacher, map_student)
        loss = loss_ce + loss_kl*self.alpha + loss_l2*self.beta

        return loss