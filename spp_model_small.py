import torch
import numpy as np
from resnet_small import resnet18
from pyramidpooling import SpatialPyramidPooling
from torchinfo import summary


class MycnnSPPNetOri(torch.nn.Module):
  def __init__(self):
    super(MycnnSPPNetOri, self).__init__()

    self.spp_layers_num = [6, 3, 2, 1]
    resnet = resnet18(pretrained=False)
    self.backbone = torch.nn.Sequential(*(list(resnet.children())[:-3]))
    self.spp_pooling = SpatialPyramidPooling(self.spp_layers_num)
    self.dropout = torch.nn.Dropout(0.5)
    self.fc_steer = torch.nn.Linear(128*np.sum(np.square(self.spp_layers_num)), 1)
    self.fc_coll = torch.nn.Linear(128*np.sum(np.square(self.spp_layers_num)), 1)
    self.activation_coll = torch.nn.Sigmoid()

  def forward(self, x):
    x = self.backbone(x)
    x = self.spp_pooling(x)
    x = self.dropout(x)

    steer = self.fc_steer(x)
    coll = self.activation_coll(self.fc_coll(x))
    return steer, coll

if __name__ == "__main__":
  cnn_ori = MycnnSPPNetOri()
  summary(cnn_ori, (1, 3, 448, 448))
