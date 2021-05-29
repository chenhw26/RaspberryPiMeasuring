import torch
import spp_model_small
import time

if __name__ == "__main__":
    # cnn_ori = spp_model.MycnnSPPNetOri().cpu()
    # cnn_ori.load_state_dict(torch.load("saved_models/cnn_spp.pt", map_location=torch.device("cpu")))
    cnn_ori = spp_model_small.MycnnSPPNetOri()
    cnn_ori.eval()

    # bottlenet = spp_model_small.MycnnBottlenetDronePart(2).cpu()
    # bottlenet.eval()

    for res in [112, 224, 448]:
        img = torch.rand(1, 3, res, res)
        for _ in range(10):
            _ = cnn_ori(img)

    print("Local compute latency:")
    for res in [112, 224, 448]:
        img = torch.rand(1, 3, res, res)
        start = time.time()
        for _ in range(50):
            timer = time.time()
            _ = cnn_ori(img)
            timer_end = time.time()
            if timer_end-timer < 0.2: time.sleep(0.2-timer_end+timer)
        end = time.time()
        print("Res {}: {:.5f}s/img".format(res, (end-start)/50))

    # print("\nBottlenet compute latency:")
    # for res in [112, 224, 448]:
    #     img = transform(load_img("test_img.jpg", crop_size=(448, 448), target_size=(res, res)))
    #     img = torch.unsqueeze(img, 0)
    #     start = time.time()
    #     for _ in range(100):
    #         _ = bottlenet(img)
    #     end = time.time()
    #     print("Res {}: {:.2f}ms/img".format(res, (end-start)*10))

    # bottlenet_server = spp_model_small.MycnnBottlenetServerPart(2)
    # bottlenet_server.eval()
    #
    # print("\nServer bottlenet compute latency:")
    # for res in [112, 224, 448]:
    #     img = transform(load_img("test_img.jpg", crop_size=(448, 448), target_size=(res, res)))
    #     img = torch.unsqueeze(img, 0)
    #     inputs = bottlenet(img)
    #     start = time.time()
    #     for _ in range(100):
    #         _ = bottlenet_server(inputs)
    #     end = time.time()
    #     print("Res {}: {:.2f}ms/img".format(res, (end-start)*10))