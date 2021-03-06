import torch
import zmq
import time
import numpy as np
import cv2
from img_utils import load_img
import pickle
import zlib

if __name__ == "__main__":
    context = zmq.Context()
    skt_send_img = context.socket(zmq.PUB)
    skt_send_img.bind("tcp://192.168.1.216:5556")  # 本机ip

    skt_warmup = context.socket(zmq.PUB)
    skt_warmup.bind("tcp://192.168.1.216:5557")  # 本机ip
    for _ in range(10):
        time.sleep(0.05)
        skt_warmup.send("ss".encode("utf-8"))

    skt_recv_ack = context.socket(zmq.SUB)
    skt_recv_ack.connect("tcp://192.168.1.161:5555")  # 远程ip
    skt_recv_ack.set(zmq.SUBSCRIBE, b"")

    # bottlenet = spp_model_small.MycnnBottlenetDronePart(2).cpu()
    # bottlenet.eval()
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    for res in [448]:
        print("Res: {}".format(res))
        img = load_img("test_img.jpg", crop_size=(448, 448), target_size=(res, res))
        # img = torch.unsqueeze(transform(img), 0)
        # data = bottlenet(img).detach().numpy()
        # data = zlib.compress(pickle.dumps(data, -1))
        img_encode = cv2.imencode(".jpg", np.array(img))[1]
        data = np.array(img_encode).tostring()
        start = time.time()
        for i in range(1000):
            print("send {} pic".format(i))
            skt_send_img.send(data)
        _ = skt_recv_ack.recv()
        end = time.time()
        print("Latency per img: {:.5f}s".format((end-start)/100.0))