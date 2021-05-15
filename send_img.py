import zmq
from image_utils import load_img
import time
import numpy as np
import cv2

if __name__ == "__main__":
    context = zmq.Context()
    skt_send_img = context.socket(zmq.PUB)
    skt_send_img.bind("tcp://192.168.1.141:5556")  # 本机ip

    skt_warmup = context.socket(zmq.PUB)
    skt_warmup.bind("tcp://192.168.1.141:5557")  # 本机ip
    for _ in range(10):
        time.sleep(0.05)
        skt_warmup.send("ss".encode("utf-8"))

    skt_recv_ack = context.socket(zmq.SUB)
    skt_recv_ack.connect("tcp://192.168.1.161:5555")  # 远程ip
    skt_recv_ack.set(zmq.SUBSCRIBE, b"")

    for res in [448]:
        print("Res: {}".format(res))
        img = load_img("test_img.jpg", crop_size=(448, 448), target_size=(res, res))
        img_encode = cv2.imencode(".jpg", np.array(img))[1]
        data = np.array(img_encode).tostring()
        start = time.time()
        for i in range(100):
            print("send {} pic".format(i))
            skt_send_img.send(data)
        _ = skt_recv_ack.recv()
        end = time.time()
        print("Latency per img: {:.5f}s".format((end-start)/100.0))