import zmq
import time

if __name__ == "__main__":
    context = zmq.Context()

    skt_recv_img = context.socket(zmq.SUB)
    skt_recv_img.connect("tcp://127.0.0.1:5556")  # 远程ip
    skt_recv_img.set(zmq.SUBSCRIBE, b"")

    skt_warmup = context.socket(zmq.SUB)
    skt_warmup.connect("tcp://127.0.0.1:5557")  # 远程ip
    skt_warmup.set(zmq.SUBSCRIBE, b"")

    skt_send_ack = context.socket(zmq.PUB)
    skt_send_ack.bind("tcp://127.0.0.1:5555") # 本机ip

    _ = skt_warmup.recv()
    for _ in range(1):
        for i in range(100):
            _ = skt_recv_img.recv()
            time.sleep(0.1)
            print("recv {} pic".format(i))
        time.sleep(0.01)
        for _ in range(10):
            time.sleep(0.05)
            skt_send_ack.send("ok".encode("utf-8"), flags=zmq.NOBLOCK)
