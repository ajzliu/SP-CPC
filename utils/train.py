from distutils.log import error
from logging.handlers import DatagramHandler
from struct import unpack
import torch
import logging
import os
import torch.nn.functional as F
import sys

## Get the same logger from main"
logger = logging.getLogger("scpc")

def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '-model_best.pth')
    
    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))

def train_multicounty(step, experiment, model, data_loader, device, optimizer, epoch, timestep, log_interval, batch_size):
    torch.autograd.set_detect_anomaly(True)
    if experiment:
        with experiment.train():
            model.train()
            total_loss = {i: 0.0 for i in range(1, timestep + 1)}
            total_acc = {i: 0.0 for i in range(1, timestep + 1)}
            for batch_idx, data in enumerate(data_loader):
                optimizer.zero_grad()
                # Transfer data to device
                for county in data:
                    data[county] = data[county].to(device)
                
                output = model(data)
                acc = torch.mean(output[1], 0)
                loss = torch.mean(output[0], 0)
                step += 1
                for i, (a, l) in enumerate(zip(acc, loss)):
                    total_loss[i+1] += l.detach().item()
                    total_acc[i+1] += a.detach().item()
                    experiment.log_metrics({'loss_{}'.format(i+1): total_loss[i+1]/(batch_idx+1),
                                                'acc_{}'.format(i+1): total_acc[i+1]/(batch_idx+1)},
                                                step = step)
                experiment.log_metrics({'loss': sum(total_loss.values())/(batch_idx+1),
                                            'acc': sum(total_acc.values())/(batch_idx+1)},
                                            step = step)
                loss.sum().backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                        epoch, batch_idx * batch_size, len(data_loader) * batch_size,
                        100. * batch_idx / len(data_loader), acc.sum().detach().item(), loss.sum().detach().item()))
            # average loss # average acc
            final_acc = sum(total_acc.values())/len(data_loader)
            final_loss = sum(total_loss.values())/len(data_loader)
            logger.info('===> Training set: Average loss: {:.4f}\tAccuracy: {:.4f}'.format(
                        final_loss, final_acc))
    else:
        model.train()
        total_loss = {i: 0.0 for i in range(1, timestep + 1)}
        total_acc = {i: 0.0 for i in range(1, timestep + 1)}
        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            # Transfer data to device
            for county in data:
                data[county] = data[county].to(device)
            output = model(data)
            acc = torch.mean(output[1], 0)
            loss = torch.mean(output[0], 0)
            step += 1
            for i, (a, l) in enumerate(zip(acc, loss)):
                total_loss[i+1] += l.detach().item()
                total_acc[i+1] += a.detach().item()
            loss.sum().backward()

            optimizer.step()
            if batch_idx % log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                    epoch, batch_idx * batch_size, len(data_loader) * batch_size,
                    100. * batch_idx / len(data_loader), acc.sum().detach().item(), loss.sum().detach().item()))
        # average loss # average acc
        final_acc = sum(total_acc.values())/len(data_loader)
        final_loss = sum(total_loss.values())/len(data_loader)
        logger.info('===> Training set: Average loss: {:.4f}\tAccuracy: {:.4f}'.format(
                    final_loss, final_acc))
    
    return final_acc, final_loss, step