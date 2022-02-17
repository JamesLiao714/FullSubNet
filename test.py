import os
import time
import torch
import shutil
import numpy as np
import config as cfg
from models import DCCRN, CRN, FullSubNet  # you can import 'DCCRN' or 'CRN' or 'FullSubNet'
from write_on_tensorboard import Writer
from dataloader import create_dataloader
from trainer import model_train, model_validate, \
    model_perceptual_train, model_perceptual_validate, \
    dccrn_direct_train, dccrn_direct_validate, \
    crn_direct_train, crn_direct_validate, \
    fullsubnet_test, model_test
import soundfile as sf


 

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
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
total_params = calculate_total_params(model)

# Set trainer and estimator
if cfg.perceptual is True:
    estimator = model_perceptual_validate
elif cfg.model == 'FullSubNet':
    estimator = fullsubnet_test
elif cfg.masking_mode == 'Direct(None make)' and cfg.model == 'DCCRN':
    estimator = dccrn_direct_validate
elif cfg.masking_mode == 'Direct(None make)' and cfg.model == 'CRN':
    estimator = crn_direct_validate
else:
    estimator = model_test
###############################################################################
#                              Create Dataloader                              #
###############################################################################

test_loader = create_dataloader(mode='test')

###############################################################################
#                        Set a log file to store progress.                    #
#               Set a hps file to store hyper-parameters information.         #
###############################################################################
print('Loading model from checkpoint: %s' % cfg.chkpt_model_test)

# Set a log file to store progress.
test_model_path = cfg.job_dir + cfg.chkpt_model_test

checkpoint = torch.load(test_model_path)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch_start_idx = checkpoint['epoch'] + 1

# if the loaded length is shorter than I expected, extend the length

# make the file directory
if not os.path.exists(cfg.result_path):
    os.mkdir(cfg.result_path)

###############################################################################
#                                    Test                                     #
###############################################################################


start_time = time.time()

# Validation
estimator(model, test_loader, DEVICE)

print('Test Resport | takes {:.2f} seconds\n'
        .format(time.time() - start_time))

print('Testing has been finished.')


