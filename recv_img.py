import zmq

if __name__ == "__main__":
    context = zmq.Context()

    skt_recv_img = context.socket(zmq.SUB)
    skt_recv_img.connect("tcp://127.0.0.1:55556")  # 远程ip
    skt_recv_img.set(zmq.SUBSCRIBE, b"")

    skt_send_ack = context.socket(zmq.PUB)
    skt_send_ack.bind("tcp://127.0.0.1:55555") # 本机ip

    skt_send_ack.send("begin".encode("utf-8"))
    for i in range(100):
        print("recv {} pic".format(i))
        _ = skt_recv_img.recv()
    skt_send_ack.send("ok".encode("utf-8"))
