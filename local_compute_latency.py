import torch
import torchvision
import spp_model
import time
from image_utils import load_img

if __name__ == "__main__":
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    cnn_ori = spp_model.MycnnSPPNetOri()
    cnn_ori.load_state_dict(torch.load("saved_models/cnn_spp.pt"))
    cnn_ori.eval()

    for res in [112, 224, 448]:
        img = transform(load_img("test_img.jpg", crop_size=(448, 448), target_size=(res, res)))
        img = torch.unsqueeze(img, 0)
        start = time.time()
        for _ in range(100):
            _ = cnn_ori(img)
        end = time.time()
        print("Res {}: {:.2f}ms/img".format(res, (end-start)*10))