import sys, os
import torch
import numpy as np

# Changable configs
models_path = {
    "A": "/data/zzp1012/disagreement/outs/disagreement/0718/0718-154034-cifar10-vgg11-iters10000-lr0.01-bs128-seed0+0+0-midpoint1000/exp/model_init.pt",
    "B": "/data/zzp1012/disagreement/outs/disagreement/0718/0718-154401-cifar10-vgg11-iters10000-lr0.01-bs128-seed0+0+1-midpoint1000/exp/model_init.pt",
}
DATASET = "cifar10"
MODEL_TYPE = "vgg11"

# import internal libs
sys.path.insert(1, os.path.join(os.path.dirname(models_path["A"]), "../src/"))
from model import prepare_model

# prepare the model
models = dict()
for key, path in models_path.items():
    model = prepare_model(MODEL_TYPE, DATASET)
    model.load_state_dict(torch.load(path))
    models[key] = model

for key, model in models.items():
    if key == "A":
        continue
    for name, param in model.named_parameters():
        print(f"{key} {name} {param.shape}")
        baseline = models["A"].state_dict()[name].cpu().detach().numpy()
        current = param.cpu().detach().numpy()
        print(f"{key} {name} {np.mean(np.abs(baseline - current))}")
        assert np.mean(np.abs(baseline - current)) < 1e-6