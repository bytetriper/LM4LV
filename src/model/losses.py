import torch
import torch.nn as nn
import torchvision.transforms as T

"""
Here you can play with existing vision loss or define new losses.
When defining your own loss, you should follow the calling convention of VisionLoss_Template, as this is how we call it in the main process.
NOTE: you don't have to inherit VisionLoss_Template. I didn't do it.
See and follow the existing loss implementations for how you can find the vision embeds in the sequence.
"""
class VisionLoss_Template(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
    
    def forward(self,feat:torch.Tensor,vision_instances:list[dict],vision_pils, target_mask:torch.Tensor, decoder:nn.Module,adapter:nn.Module )->torch.Tensor:
        raise NotImplementedError
class VQ_CrossEntropyLoss(nn.Module):
    def __init__(self,ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    def forward(self,feat:torch.Tensor,vision_instances:list[dict],vision_pils, target_mask:torch.Tensor, decoder:nn.Module,adapter:nn.Module )->torch.Tensor:
        total_len = len(vision_instances)
        loss = 0.
        actual_len = 0
        for k in range(total_len):
            idx = target_mask == k
            if idx.sum() == 0:
                continue
            logits_k = feat[idx]
            actual_len += 1
            logits_k_gt = vision_instances[k]
            loss += self.loss(logits_k,logits_k_gt)
        return loss/actual_len if actual_len > 0 else 0.
class Feat_MSELoss(nn.Module):
    def __init__(self,num_vision_tokens: int):
        super().__init__()
        self.loss = nn.MSELoss()
        #self.loss = nn.L1Loss()
        self.num_vision_token = num_vision_tokens
    def forward(self,feat:torch.Tensor,vision_instances:list[dict],vision_pils, target_mask:torch.Tensor, decoder:nn.Module,adapter:nn.Module )->torch.Tensor:
        total_len = len(vision_instances)
        loss = 0.
        actual_len = 0
        for k in range(total_len):
            idx = target_mask == k
            if idx.sum() == 0: # no token in this mask
                continue
            actual_len += 1
            feat_k = feat[idx]
            vision_k = vision_instances[k]
            loss += self.loss(feat_k,vision_k)
        return loss/actual_len if actual_len > 0 else 0.

class MAE_PixelLoss(nn.Module):
    def __init__(self,num_vision_tokens: int):
        super().__init__()
        self.loss = nn.L1Loss()
        self.num_vision_token = num_vision_tokens
        self.ToTensor = T.ToTensor()
    def forward(self,feat:torch.Tensor,vision_instances:list[dict],vision_pils, target_mask:torch.Tensor, decoder:nn.Module,adapter:nn.Module )->torch.Tensor:
        total_len = len(vision_instances)
        loss = 0.
        actual_len = 0
        for k in range(total_len):
            idx = target_mask == k
            if idx.sum() == 0: # no token in this mask
                continue
            actual_len += 1
            feat_k = feat[idx].unsqueeze(0)
            feat_k = decoder(feat_k)
            size = feat_k.shape[2:]
            vision_pil_k = vision_pils[k].resize(size)
            vision_k = self.ToTensor(vision_pil_k).unsqueeze(0).to(feat_k.device).to(feat_k.dtype)
            loss += self.loss(feat_k,vision_k)
        return loss/actual_len if actual_len > 0 else 0.