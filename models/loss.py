import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from IQA_pytorch import LPIPSvgg

# class mse_loss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.loss_fn = nn.MSELoss()
#     def forward(self, output, target):
#         return self.loss_fn(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)


class AuxLoss(nn.Module):
    def __init__(self, feat_coeff=5.0, pixel_coeff=100.0):
        super(AuxLoss, self).__init__()
        self.feat_loss_fn = LPIPSvgg()
        self.feat_coeff = feat_coeff
        self.pixel_coeff = pixel_coeff

    def forward(self, output, target):
        output = (output + 1) / 2.0
        target = (target + 1) / 2.0
        pixel_loss = F.mse_loss(output, target)
        output= output.repeat_interleave(3, dim=1)
        target = target.repeat_interleave(3, dim=1)
        feat_loss = self.feat_loss_fn(output, target)
        print(feat_loss, pixel_loss)
        return self.pixel_coeff * pixel_loss + self.feat_coeff * feat_loss

    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

