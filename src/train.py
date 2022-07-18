import os
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import NoReturn
from tqdm import tqdm

# import internal libs
from utils import get_logger, update_dict

def sample_indices(total_num: int,
                   sample_size: int,
                   seed: int) -> np.array:
    """sample the indices from the trainset

    Args:
        total_num: the total number
        sample_size: the sample size
        seed: the seed

    Return:
        the sampled indices
    """
    # create the indices
    indices = np.arange(total_num)
    random.Random(seed).shuffle(indices)
    return indices[:sample_size]


def eval(device: torch.device,
         model: nn.Module,
         testloader: DataLoader):
    """evaluate the model

    Args:
        device: GPU or CPU
        model: the model to evaluate
        testloader: the test dataset loader
    """
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    # evaluatioin
    model.eval()
    with torch.no_grad():
        # testset
        test_losses, test_acc = [], 0
        for inputs, labels in tqdm(testloader):
            # set the inputs to device
            inputs, labels = inputs.to(device), labels.to(device)
            # set the outputs
            outputs = model(inputs)
            # set the loss
            losses = loss_fn(outputs, labels)
            # set the loss and accuracy
            test_losses.extend(losses.cpu().detach().numpy())
            test_acc += (outputs.max(1)[1] == labels).sum().item()
    # print the test loss and accuracy
    test_loss = np.mean(test_losses)
    test_acc /= len(testloader.dataset)
    return test_loss, test_acc


def train(save_path: str,
          device: torch.device,
          model: nn.Module,
          trainset: Dataset,
          testset: Dataset,
          iters: int,
          lr: float,
          batch_size: int,
          weight_decay: float,
          momentum: float,
          fir_seed: int,
          sec_seed: int,
          midpoint: int) -> NoReturn:
    """train the model

    Args:
        save_path: the path to save results
        device: GPU or CPU
        model: the model to train
        trainset: the train dataset
        testset: the test dataset
        iters: the iterations
        lr: the learning rate
        batch_size: the batch size
        weight_decay: the weight decay
        momentum: the momentum
        fir_seed: the seed for the first phase
        sec_seed: the seed for the second phase
        the midpoint: the midpoint between the first and second phase
    """
    logger = get_logger(__name__)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ## set up the basic component for training
    # put the model to GPU or CPU
    model = model.to(device)
    # set the optimizer
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=lr, weight_decay=weight_decay, momentum=momentum)
    # set the loss function
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    
    ## set up the data part
    # load the whole set of trainset
    t_inputs, t_labels = next(iter(DataLoader(trainset, batch_size=len(trainset))))   
    # set the trainset loader
    trainloader = DataLoader(trainset, batch_size=batch_size)
    # set the testset loader 
    testloader = DataLoader(testset, batch_size=batch_size)
    
    # create the seeds for the first phase
    fir_seeds = random.Random(fir_seed).sample(range(10000000), k=midpoint)
    sec_seeds = random.Random(sec_seed).sample(range(10000000), k=iters - midpoint)
    seeds = fir_seeds + sec_seeds
    
    # save the initial model
    torch.save(model.state_dict(), os.path.join(save_path, f"model_init.pt"))

    # initialize the res_dict
    total_res_dict = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    # train the model
    for itr in range(1, iters+1):
        # create the batches for train
        sampled_indices = sample_indices(len(trainset), batch_size, seeds[itr-1])
        # train the model
        model.train()
        # get the inputs and labels
        inputs, labels = t_inputs[sampled_indices], t_labels[sampled_indices]
        # set the inputs to device
        inputs, labels = inputs.to(device), labels.to(device)
        # set the outputs
        outputs = model(inputs)
        # set the loss
        losses = loss_fn(outputs, labels)
        loss = torch.mean(losses)
        # set zero grad
        optimizer.zero_grad()
        # set the loss
        loss.backward()
        # set the optimizer
        optimizer.step()

        ## evaluate the model
        if itr % 10 == 0:
            # eval on the trainset
            train_loss, train_acc = eval(device, model, trainloader)
            # eval on the testset
            test_loss, test_acc = eval(device, model, testloader)

            # update res_dict
            res_dict = {
                "train_loss": [train_loss],
                "train_acc": [train_acc],
                "test_loss": [test_loss],
                "test_acc": [test_acc],
            }
            total_res_dict = update_dict(res_dict, total_res_dict)

            # print the results
            logger.info(f"itr: {itr}, train_loss: {train_loss}, train_acc: {train_acc},\
                test_loss: {test_loss}, test_acc: {test_acc}")

        # save the results
        if itr % 100 == 0 or itr == iters:
            torch.save(model.state_dict(), 
                       os.path.join(save_path, f"model_itr{itr}.pt"))
            res_df = pd.DataFrame.from_dict(total_res_dict)
            res_df.to_csv(os.path.join(save_path, "train.csv"), index = False)