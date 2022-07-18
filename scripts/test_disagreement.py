import sys, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Changable configs
models_path = {
    "A": "/data/zzp1012/disagreement/outs/tmp/0708-011128-seed0-cifar10-vgg11-epochs100-lr0.01-bs128-wd0.0-momentum0.0/exp/model_100.pt",
    "B": "/data/zzp1012/disagreement/outs/tmp/0718-135203-seed1-cifar10-vgg11-epochs100-lr0.01-bs128-wd0.0-momentum0.0/exp/model_100.pt",
}
DATASET = "cifar10"
MODEL_TYPE = "vgg11"
BATCH_SIZE = 128
DEVICE = "cuda:0"

# import internal libs
sys.path.insert(1, os.path.join(os.path.dirname(models_path["A"]), "../src/"))
from model import prepare_model
from data import prepare_dataset

# prepare the model
models = dict()
for key, path in models_path.items():
    model = prepare_model(MODEL_TYPE, DATASET)
    model.load_state_dict(torch.load(path))
    models[key] = model.to(DEVICE)

# prepare the data
_, testset = prepare_dataset(DATASET, "../data/")
testloader = DataLoader(testset, batch_size=BATCH_SIZE)

# evaluatioin
preds_dict = dict()
for key, model in models.items():
    model.eval()
    preds_dict[key] = []
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            # set the inputs to device
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            # set the outputs
            outputs = model(inputs)
            # set the preds
            preds = outputs.max(1)[1]
            # keep the preds
            preds_dict[key].extend(preds.cpu().detach().numpy())
    preds_dict[key] = np.array(preds_dict[key])

# print the preds
for key, preds in preds_dict.items():
    print(f"{key}: {len(preds)}")
    print(preds[:10])
    print("\n")

# calculate the disagreement
disagreement = 0
for key, preds in preds_dict.items():
    disagreement += np.sum(preds == preds_dict["A"])
    disagreement /= len(preds_dict["A"])
    print(f"for model {key}: the disagreement is {disagreement}")