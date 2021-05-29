import torch
import spp_model_small
import time

if __name__ == "__main__":
    cnn_ori = spp_model_small.MycnnSPPNetOri().cuda()
    cnn_ori.eval()

    for res in [112, 224, 448]:
        img = torch.rand(1, 3, res, res).cuda()
        for _ in range(10):
            _ = cnn_ori(img)

    print("Local compute latency:")
    res = 448
    img = torch.rand(1, 3, res, res).cuda()
    while True:
        # timer = time.time()
        _ = cnn_ori(img)
        # timer_end = time.time()
        # if timer_end-timer < 0.2: time.sleep(0.2-timer_end+timer)