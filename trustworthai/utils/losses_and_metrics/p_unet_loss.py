from trustworthai.utils.losses_and_metrics.dice_loss import DiceLossWithWeightedEmptySlices
import torch
import torch.nn as nn

class punet_loss(nn.Module):
    def __init__(self, xent_factor, dice_factor, dice_sample_factor, kl_factor, dice_empty_slice_weight):
        super().__init__()
        self.dice_factor = dice_factor
        self.dice_sample_factor = dice_sample_factor
        self.xent_factor = xent_factor
        self.kl_factor = kl_factor
        self.dice_loss = DiceLossWithWeightedEmptySlices(dice_empty_slice_weight)
        
    def forward(self, net, pred_sample, pred_mean, target):
        #elbo = net.elbo(target)
        xent, kl = net.elbo(target)
        #reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)

        # print("elbo: ", elbo)

        # print("xent: " , xent)
        # print("kl: ", kl)
        # print("reg: ", reg_loss * 1e-5)

        dice_sample = self.dice_loss(pred_sample, target)
        dice_mean = self.dice_loss(pred_mean, target)
        
        #print("dice sample and mean", dice_sample, dice_mean)
        #print("factors xent kl dice mean dice sample: ", self.xent_factor, self.kl_factor, self.dice_factor, self.dice_sample_factor) 

        if pred_mean.max() > 1000 or pred_mean.min() < -1000:
            print(pred_mean.max(), pred_mean.min())
            raise ValueError("to high")

        # print("dice: ", dice)

        #loss = -elbo * 1e-3 + 1e-5 * reg_loss + dice * 30
        #loss = -elbo * 1e-3 + 1e-5 * reg_loss + dice
        #loss = -elbo * 1e-3 + reg_loss * 1e-5 + dice
        #loss = dice #* 300
        #loss = -xent *2e-4 -kl +1e-5*reg_loss + dice * dice_factor
        #loss = dice * 50 - kl
        #loss = dice
        #loss = -kl +(1e-5*reg_loss) + dice * 50
        loss = (-xent * self.xent_factor) + (-kl*self.kl_factor) + (dice_mean*self.dice_factor) + (dice_sample*self.dice_factor*self.dice_sample_factor)
    
        return loss