import time
import torch
import multiprocessing
from spp_model_small import MycnnSPPNetOri

def process_func(identifier):
  inputs = torch.rand((1, 3, 448, 448))
  model = MycnnSPPNetOri()
  model.eval()
  print("id: {}, start".format(identifier))
  start = time.time()
  for _ in range(100):
    _ = model(inputs)
  end = time.time()
  print("id: {}, {:.3f}s/img".format(
    identifier, (end-start)/100))

if __name__ == "__main__":
  processes = []
  for i in range(1):
    processes.append(multiprocessing.Process(
      target=process_func, args=(i, )))
  for process in processes:
    process.start()
  for process in processes:
    process.join()