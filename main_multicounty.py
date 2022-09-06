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

## Torch
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim

## Custom Imports
from utils.logger import setup_logs
from utils.seed import set_seed
from utils.train import snapshot, train_multicounty
from utils.validation import validation_multicounty
from utils.dataset_overlap_scpc import COVIDCounties, MobilityData
from model.models import SCPC

import sys

has_comet = False

############ Control Center and Hyperparameter ###############
config = box_from_file(Path('config.yaml'), file_type='yaml')
if config.training.resume_name:
    run_name = config.training.resume_name
else:
    run_name = "scpc" + time.strftime("-%Y-%m-%d_%H_%M_%S")
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
                                         project_name="scpc-covid-1")
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
        
# create a CPC model for NLP
mobility = MobilityData(config)
model = SCPC(config=config, weights=mobility.data)

# load model if resume mode
if config.training.resume_name:
    logger.info('===> loading a checkpoint')
    checkpoint = torch.load('{}/{}-{}'.format(config.training.logging_dir, run_name,'model_best.pth'))
    model.load_state_dict(checkpoint['state_dict'])
# line for multi-gpu
if config.training.multigpu and torch.cuda.device_count() > 1:
    logger.info("===> let's use {} GPUs!".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
# move to device
model.to(device)

## Loading the dataset
logger.info('===> loading train and validation dataset')
dataset = COVIDCounties(config)

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
    np.random.shuffle(indices)

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

# for batch in zip(*train_loaders.values()):
#     # This creates a list that contains a batch from each of the counties.
#     # However, it does not label the batches in the list with their
#     # respective counties.

#     # Because zip() zips together the iterables in the order that they were
#     # listed in the parameters, and values() returns the values in the dict
#     # in the order they were added. (from Python 3.6 onwards, so make sure
#     # that you're using 3.6 or beyond)

#     # This is unreliable, so if you know of a better solution, please let me know :)
    
#     # Zips together the county FIPS codes and the batches and then converts to dict
#     batch = dict(zip(dataset.counties, batch))

# Generator comprehension version of the above code
combined_train_loader = [dict(zip(dataset.counties, batch)) for batch in zip(*train_loaders.values())]
combined_validation_loader = [dict(zip(dataset.counties, batch)) for batch in zip(*validation_loaders.values())]

# Section 3.3: Adam optimizer with a learning rate of 2e-4
optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-3, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True)
if config.training.resume_name:
    optimizer.load_state_dict(checkpoint['optimizer'])
    
model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info('### Model summary below ###\n {}'.format(str(model)))
logger.info('===> Model total parameter: {}\n'.format(model_params))
if has_comet: experiment.set_model_graph(str(model))

## Start training
if config.training.resume_name:
    best_acc = checkpoint['validation_acc']
    best_loss = checkpoint['validation_loss']
    best_epoch = checkpoint['epoch']
    step = checkpoint['step_train']
    initial_epoch = checkpoint['epoch']
else:
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1 
    step = 0
    initial_epoch = 1

logger.info('### Training begins at epoch {} and step {} ###'.format(initial_epoch,step))
for epoch in range(initial_epoch, config.training.epochs + 1):
    epoch_timer = timer()
    # Train and validate
    _, _, step = train_multicounty(
        step, 
        experiment, 
        model, 
        combined_train_loader, 
        device, 
        optimizer, 
        epoch, 
        config.scpc_model.k_size, 
        config.training.log_interval,
        batch_size)
    val_acc, val_loss = validation_multicounty(
        step, 
        experiment, 
        model, 
        combined_validation_loader, 
        device, 
        config.scpc_model.k_size)     
    # Save
    if val_acc > best_acc: 
        best_acc = max(val_acc, best_acc)
        if torch.cuda.device_count() > 1:
            dict_to_save = model.module.state_dict()
        else:
            dict_to_save = model.state_dict()
        snapshot(config.training.logging_dir, run_name, {
            'epoch': epoch,
            'step_train': step,
            'validation_acc': val_acc,
            'validation_loss': val_loss,
            'state_dict': dict_to_save,
            'optimizer': optimizer.state_dict(),
        })
        best_epoch = epoch
    end_epoch_timer = timer()
    logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, config.training.epochs, end_epoch_timer - epoch_timer))

## end 
end_global_timer = timer()
logger.info("################## Success #########################")
logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))