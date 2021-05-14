import zmq
from image_utils import load_img
import time
import numpy as np
import cv2

if __name__ == "__main__":
    context = zmq.Context()
    skt_send_img = context.socket(zmq.PUB)
    skt_send_img.bind("tcp://127.0.0.1:55556")  # 本机ip

    skt_recv_ack = context.socket(zmq.SUB)
    skt_recv_ack.connect("tcp://127.0.0.1:55555")  # 远程ip
    skt_recv_ack.set(zmq.SUBSCRIBE, "".encode("utf-8"))

    print("Wait for ack")
    _ = skt_recv_ack.recv()
    for res in [112, 224, 448]:
        print("Res: {}".format(res))
        img = load_img("test_img.jpg", crop_size=(448, 448), target_size=(res, res))
        img_encode = cv2.imencode(".jpg", img)[1]
        data = np.array(img_encode).tostring()
        start = time.time()
        for i in range(100):
            print("send {} pic".format(i))
            skt_send_img.send(data, zmq.NOBLOCK)
        _ = skt_recv_ack.recv()
        end = time.time()
        print("Latency per img: {:.5f}s".format((end-start)/100.0))