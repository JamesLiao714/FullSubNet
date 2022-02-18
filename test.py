import os
import time
import torch
import shutil
import numpy as np
import config as cfg
from models import DCCRN, CRN, FullSubNet  # you can import 'DCCRN' or 'CRN' or 'FullSubNet'
from dataloader import create_dataloader
from trainer import  fullsubnet_test, model_test



 

###############################################################################
#                        Helper function definition                           #
###############################################################################


# Calculate the size of total network.
def calculate_total_params(our_model):
    total_parameters = 0
    for variable in our_model.parameters():
        shape = variable.size()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    return total_parameters

###############################################################################
#         Parameter Initialization and Setting for model training             #
###############################################################################
# Set device
DEVICE = torch.device(cfg.DEVICE)

# Set model
if cfg.model == 'DCCRN':
    model = DCCRN().to(DEVICE)
elif cfg.model == 'CRN':
    model = CRN().to(DEVICE)
elif cfg.model == 'FullSubNet':
    model = FullSubNet().to(DEVICE)
# Set optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr = cfg.learning_rate)
total_params = calculate_total_params(model)

# Set estimator
if cfg.model == 'FullSubNet':
    estimator = fullsubnet_test
else:
    print('NO test module')

# Create Dataloader     
test_loader = create_dataloader(mode='test')


#  Set a log file to store progress.   
#  Set a hps file to store hyper-parameters information.    

print('Loading model from checkpoint: %s' % cfg.chkpt_model_test)

# Set a log file to store progress.
test_model_path = cfg.job_dir + cfg.chkpt_model_test

checkpoint = torch.load(test_model_path)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch_start_idx = checkpoint['epoch'] + 1

# make the file directory
if not os.path.exists(cfg.result_path):
    os.mkdir(cfg.result_path)

#  Test                                


start_time = time.time()

# test, generate result
estimator(model, test_loader, DEVICE)

print('Test Resport | takes {:.2f} seconds\n'
        .format(time.time() - start_time))

print('Testing has been finished.')


