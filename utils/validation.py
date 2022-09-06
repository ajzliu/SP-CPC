import numpy as np
import logging
import torch
import torch.nn.functional as F

## Get the same logger from main"
logger = logging.getLogger("scpc")

def validation_multicounty(step, experiment, model, data_loader, device, timestep):
    if experiment:
        with experiment.validate():
            logger.info("Starting Validation")
            model.eval()
            total_loss = {i: 0.0 for i in range(1, timestep + 1)}
            total_acc = {i: 0.0 for i in range(1, timestep + 1)}
            with torch.no_grad():
                for batch_idx, data in enumerate(data_loader):
                    # Transfer data to device
                    for county in data:
                        data[county] = data[county].to(device)

                    output = model(data)
                    acc = torch.mean(output[1], 0)
                    loss = torch.mean(output[0], 0)
                    for i, (a, l) in enumerate(zip(acc, loss)):
                        total_loss[i+1] += l.detach().item()
                        total_acc[i+1] += a.detach().item()

            # average loss # average acc
            final_acc = sum(total_acc.values())/len(data_loader)
            final_loss = sum(total_loss.values())/len(data_loader)
            experiment.log_metrics({'loss': final_loss,
                                        'acc': final_acc},
                                        step = step)
            logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                        final_loss, final_acc))
    else:
        logger.info("Starting Validation")
        model.eval()
        total_loss = {i: 0.0 for i in range(1, timestep + 1)}
        total_acc = {i: 0.0 for i in range(1, timestep + 1)}
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                # Transfer data to device
                for county in data:
                    data[county] = data[county].to(device)
                
                output = model(data)
                acc = torch.mean(output[1], 0)
                loss = torch.mean(output[0], 0)
                for i, (a, l) in enumerate(zip(acc, loss)):
                    total_loss[i+1] += l.detach().item()
                    total_acc[i+1] += a.detach().item()

        # average loss # average acc
        final_acc = sum(total_acc.values())/len(data_loader)
        final_loss = sum(total_loss.values())/len(data_loader)
        logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                    final_loss, final_acc))
            
    return final_acc, final_loss