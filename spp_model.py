import torch
import numpy as np
from resnet import resnet34
from pyramidpooling import SpatialPyramidPooling
from torchinfo import summary


class MycnnSPPNetOri(torch.nn.Module):
  def __init__(self):
    super(MycnnSPPNetOri, self).__init__()

    self.spp_layers_num = [6, 3, 2, 1]
    resnet = resnet34(pretrained=False)
    self.backbone = torch.nn.Sequential(*(list(resnet.children())[:-3]))
    self.spp_pooling = SpatialPyramidPooling(self.spp_layers_num)
    self.dropout = torch.nn.Dropout(0.5)
    self.fc_steer = torch.nn.Linear(256*np.sum(np.square(self.spp_layers_num)), 1)
    self.fc_coll = torch.nn.Linear(256*np.sum(np.square(self.spp_layers_num)), 1)
    self.activation_coll = torch.nn.Sigmoid()

  def forward(self, x):
    x = self.backbone(x)
    x = self.spp_pooling(x)
    x = self.dropout(x)

    steer = self.fc_steer(x)
    coll = self.activation_coll(self.fc_coll(x))
    return steer, coll


class MycnnSPPNetFront(torch.nn.Module):
  def __init__(self):
    super(MycnnSPPNetFront, self).__init__()

    ori_spp = MycnnSPPNetOri()
    ori_spp.load_state_dict(torch.load("saved_models/cnn_spp.pt"))
    backbone = list(ori_spp.children())[0]
    self.backbone_front = torch.nn.Sequential(*(list(backbone.children())[:-2]))

  def forward(self, x):
    x = self.backbone_front(x)
    return x


class MycnnSPPNetBack(torch.nn.Module):
  def __init__(self):
    super(MycnnSPPNetBack, self).__init__()

    ori_spp = MycnnSPPNetOri()
    ori_spp.load_state_dict(torch.load("saved_models/cnn_spp.pt"))
    backbone = list(ori_spp.children())[0]
    self.backbone_back = torch.nn.Sequential(*(list(backbone.children())[-2:]))
    self.spp_pooling = list(ori_spp.children())[1]
    self.dropout = list(ori_spp.children())[2]
    self.fc_steer = list(ori_spp.children())[3]
    self.fc_coll = list(ori_spp.children())[4]
    self.activation = list(ori_spp.children())[5]

  def forward(self, x):
    x = self.backbone_back(x)
    x = self.spp_pooling(x)
    x = self.dropout(x)

    steer = self.fc_steer(x)
    coll = self.activation(self.fc_coll(x))
    return steer, coll


class MycnnSPPNetFrontWithBottlenect(torch.nn.Module):
  def __init__(self, bottleneck_channel):
    super(MycnnSPPNetFrontWithBottlenect, self).__init__()

    self.extractor = torch.nn.Sequential(
      torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
      torch.nn.BatchNorm2d(64),
      torch.nn.ReLU(inplace=True),
      torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    self.bn = torch.nn.Sequential(
      torch.nn.BatchNorm2d(64),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
      torch.nn.BatchNorm2d(bottleneck_channel),
      torch.nn.ReLU(inplace=True),
      torch.nn.ConvTranspose2d(bottleneck_channel, 256, kernel_size=4, stride=2, bias=False),
      torch.nn.BatchNorm2d(256),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(256, 128, kernel_size=2, stride=1, bias=False),
      torch.nn.BatchNorm2d(128),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(128, 128, kernel_size=2, stride=1, bias=False),
      torch.nn.BatchNorm2d(128),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(128, 64, kernel_size=2, stride=1, bias=False),
      torch.nn.AvgPool2d(kernel_size=2, stride=1)
    )

  def forward(self, x):
    x = self.extractor(x)
    x = self.bn(x)
    return x


class MycnnBottlenetDronePart(torch.nn.Module):
  def __init__(self, bt_channel):
    super(MycnnBottlenetDronePart, self).__init__()
    sppnet_front_bottlenet = MycnnSPPNetFrontWithBottlenect(bt_channel)
    checkpoint = torch.load("saved_models/checkpoint_bottleneck_28")
    sppnet_front_bottlenet.load_state_dict(checkpoint["model_state_dict"])

    self.extractor = list(sppnet_front_bottlenet.children())[0]
    self.compressor = list(list(sppnet_front_bottlenet.children())[1].children())[:5]
    self.compressor = torch.nn.Sequential(*self.compressor)

  def forward(self, x):
    x = self.extractor(x)
    x = self.compressor(x)
    return x


class MycnnBottlenetServerPart(torch.nn.Module):
  def __init__(self, bt_channel):
    super(MycnnBottlenetServerPart, self).__init__()
    sppnet_front_bottlenet = MycnnSPPNetFrontWithBottlenect(bt_channel)
    checkpoint = torch.load("saved_models/checkpoint_bottleneck_28")
    sppnet_front_bottlenet.load_state_dict(checkpoint["model_state_dict"])
    self.decompressor = list(list(sppnet_front_bottlenet.children())[1].children())[5:]
    self.decompressor = torch.nn.Sequential(*self.decompressor)

    self.sppnet_back = MycnnSPPNetBack()

  def forward(self, x):
    x = self.decompressor(x)
    x = self.sppnet_back(x)
    return x


if __name__ == "__main__":
  # spp_net_front = MycnnSPPNetFront()
  # print(spp_net_front)
  # bottlenect = MycnnSPPNetFrontWithBottlenect(2)
  # spp_net_back = MycnnSPPNetBack()
  cnn_ori = MycnnSPPNetOri()
  summary(cnn_ori, (1, 3, 448, 448))
  # bottlenet_drone = MycnnBottlenetDronePart(2)
  # bottlenet_server = MycnnBottlenetServerPart(2)
  #
  # summary(bottlenet_server, (1, 2, 57, 57))
  # summary(bottlenet_server, (1, 2, 43, 43))
  # summary(bottlenet_server, (1, 2, 29, 29))
  # summary(bottlenet_server, (1, 2, 15, 15))