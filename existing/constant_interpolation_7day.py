import torch
from torch.utils import data
import numpy as np
from box import box_from_file
from pathlib import Path
import yaml
import warnings

## Custom Imports
from utils.seed import set_seed
from utils.dataset_nonoverlap import COVIDCounties, MobilityData

warnings.filterwarnings("ignore")
config = box_from_file(Path('config.yaml'), file_type='yaml')
use_cuda = not config.training.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# set seed for reproducibility
set_seed(config.training.seed, use_cuda)

dataset = COVIDCounties(config)

if torch.cuda.is_available():
    batch_size = int(32*torch.cuda.device_count())
else:
    batch_size = int(32)

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

per_diffs = []
with torch.no_grad():
    for batch_idx, data in enumerate(combined_train_loader):
        # Transfer data to device
        for county in data:
            data[county] = data[county].to(device)
        
        for county in data:
            to_iter = data[county]
            for bt in to_iter:
                data_in = bt[:3]
                data_in_np = data_in.cpu().numpy()
                all_pred_vals = data_in_np[2]
                targets = bt[3]
                targets_np = targets.cpu().numpy()
                for val_ct, pred_val in enumerate(all_pred_vals):
                    comp_val = targets_np[val_ct]
                    if(comp_val >= 100):
                        abs_percent_diff = abs((pred_val - comp_val) / comp_val)
                        per_diffs.append(abs_percent_diff)

print("Training set: " + str(sum(per_diffs) / len(per_diffs)))

per_diffs = []
with torch.no_grad():
    for batch_idx, data in enumerate(combined_validation_loader):
        # Transfer data to device
        for county in data:
            data[county] = data[county].to(device)
        
        for county in data:
            to_iter = data[county]
            for bt in to_iter:
                data_in = bt[:3]
                data_in_np = data_in.cpu().numpy()
                all_pred_vals = data_in_np[2]
                targets = bt[3]
                targets_np = targets.cpu().numpy()
                for val_ct, pred_val in enumerate(all_pred_vals):
                    comp_val = targets_np[val_ct]
                    if(comp_val >= 100):
                        abs_percent_diff = abs((pred_val - comp_val) / comp_val)
                        per_diffs.append(abs_percent_diff)

print("Validation set: " + str(sum(per_diffs) / len(per_diffs)))