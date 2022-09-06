## Torch
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim

## Utilities
try:
    import comet_ml
    has_comet = True
except (ImportError):
    has_comet = False

import time
import os
import logging
import yaml
from timeit import default_timer as timer

## Libraries
import numpy as np
from box import box_from_file
from pathlib import Path

## Other
import matplotlib.pyplot as plt

## Custom Imports
from utils.logger import setup_logs
from utils.seed import set_seed
from utils.dataset_overlap_scpc import COVIDCounties, MobilityData
from model.models import SCPC

has_comet = False

############ Control Center and Hyperparameter ###############
config = box_from_file(Path('config.yaml'), file_type='yaml')
if config.training.resume_name:
    run_name = config.training.resume_name
else:
    run_name = "metropolis_hastings" + time.strftime("-%Y-%m-%d_%H_%M_%S")
# setup logger    
global_timer = timer() # global timer
logger = setup_logs(config.training.logging_dir, run_name) # setup logs
logger.info('### Experiment {} ###'.format(run_name))
logger.info('### Hyperparameter summary below ###\n {}'.format(config))
# setup of comet_ml
if has_comet:
    logger.info('### Logging with comet_ml ###')
    if config.comet.previous_experiment:
        logger.info('===> using existing experiment: {}'.format(config.comet.previous_experiment))
        experiment = comet_ml.ExistingExperiment(api_key=config.comet.api_key,
                                                 previous_experiment=config.comet.previous_experiment)    
    else:
        logger.info('===> starting new experiment')
        experiment = comet_ml.Experiment(api_key=config.comet.api_key,
                                         project_name="metropolis-hastings")
    experiment.set_name(run_name)
    experiment.log_parameters({**config.training.to_dict() , 
                               **config.dataset.to_dict() , 
                               **config.scpc_model.to_dict()})
else:
    experiment = None

# define if gpu or cpu
use_cuda = not config.training.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
logger.info('===> use_cuda is {}'.format(use_cuda))
# set seed for reproducibility
set_seed(config.training.seed, use_cuda)

## Loading the dataset
logger.info('===> loading train and validation dataset')
dataset = COVIDCounties(config)
mobility = MobilityData(config)
model = SCPC(config=config, weights=mobility.data)

model_pth = 'final_models/spatial-30counties-dim60.pth'
checkpoint = torch.load(model_pth)
model.load_state_dict(checkpoint['state_dict'])

# line for multi-gpu
if config.training.multigpu and torch.cuda.device_count() > 1:
    logger.info("===> let's use {} GPUs!".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
# move to device
model.to(device)

if torch.cuda.is_available():
    batch_size = int(config.training.batch_size*torch.cuda.device_count())
else:
    batch_size = int(config.training.batch_size)

validation_split = 0.2 # 20% of dataset for validation
train_loaders = {}
validation_loaders = {}

for county in dataset.data:
    # split to train val
    dataset_size = len(dataset.data[county])
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))   

    # Randomize indices (because the data is generated in sequential order)
    rng = np.random.default_rng(config.training.seed)
    rng.shuffle(indices)

    # get random indices (shuffle equivalent)
    train_indices, valid_indices = indices[split:], indices[:split]

    # create samplers
    train_sampler = data.sampler.SubsetRandomSampler(train_indices)
    validation_sampler = data.sampler.SubsetRandomSampler(valid_indices)

    # create loaders
    train_loaders[county] = data.DataLoader(dataset.data[county],
                               batch_size=batch_size,
                               sampler=train_sampler,
                               drop_last=True
                              )
    validation_loaders[county] = data.DataLoader(dataset.data[county], 
                                    batch_size=batch_size, 
                                    sampler=validation_sampler, 
                                    drop_last=True
                                   )

# Generator comprehension version of the above code
combined_train_loader = [dict(zip(dataset.counties, batch)) for batch in zip(*train_loaders.values())]
combined_validation_loader = [dict(zip(dataset.counties, batch)) for batch in zip(*validation_loaders.values())]

model.eval()

per_diffs = []

with torch.no_grad():
    for batch_idx, data in enumerate(combined_validation_loader):
        # Transfer data to device
        for county in data:
            data[county] = data[county].to(device)
        
        _, _, context, x = model(data)
        
        for ct in context:
            x_before = x[ct][0]
            x_t = x_before[2, :]
            ct_batch = context[ct][0]
            pred_k, pred_all, plot_arr = model.forward_MetropolisHastingsPrediction(ct_batch, x_t, ct, device)
            x_tp1 = x_before[3, :]
            x_t_np = x_t.cpu().numpy()
            x_tp1_np = x_tp1.cpu().numpy()
            logger.info("")
            logger.info('x_t_np: {}'.format(x_t_np))
            logger.info('x_tp1_np: {}'.format(x_tp1_np))
            logger.info('pred_k: {}'.format(pred_k))
            logger.info('ct: {}'.format(ct))
            logger.info("")
            for val_ct, val in enumerate(pred_k):
                comp_val = x_tp1_np[val_ct]
                if(comp_val < 100):
                    break
                abs_percent_diff = abs((val - comp_val) / comp_val)
                per_diffs.append(abs_percent_diff)
        logger.info('### BATCH INDEX: {}'.format(batch_idx))
        logger.info("---------------------------------------")

print("Percent difference: " + str(sum(per_diffs) / len(per_diffs)))
                    
# Section 3.3: Adam optimizer with a learning rate of 2e-4
optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=2e-4, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True)
if config.training.resume_name:
    optimizer.load_state_dict(checkpoint['optimizer'])
    
model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info('### Model summary below ###\n {}'.format(str(model)))
logger.info('===> Model total parameter: {}\n'.format(model_params))
if has_comet: experiment.set_model_graph(str(model))

## end 
end_global_timer = timer()
logger.info("################## Success #########################")
logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))