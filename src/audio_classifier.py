import numpy as np
np.random.seed(1001)

import os
import sys
import shutil

# import IPython
# import IPython.display as ipd  # To play sound in the notebook

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tqdm

import torch 

from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter


from src.feature import AudioDataLoader
from src.conv2d_model import Conv2D_Net
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.optim import Adam, SGD


class AudioClassifier:
    def __init__(self, config):
        self.config = config
        
        self.data_loader = AudioDataLoader(config = self.config)
        self.model = Conv2D_Net()
        
        self.loss = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(),
                              lr=self.config.learning_rate,
                              weight_decay=self.config.weight_decay
                             )
        
        ## initialize counter
        self.current_epoch = 0 
        self.current_iteration = 0
        self.best_valid_acc = 0
        
        ## TODO : set cuda flag
        
        ## TODO : loading from the latest checkpoint
        
        ## Initialize Summary Writer
        self.summary_writer = SummaryWriter(log_dir = self.config.summary_dir, comment="Audio-conv2d")
        
        ## Scheduler for the optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    'min',
                                                                    patience=self.config.learning_rate_patience,
                                                                    min_lr=1e-10,
                                                                    verbose=True)

    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def run(self):
        """
        This function will the operator
        :return:
        """
        assert self.config.mode in ['train', 'test']
        try:
            if self.config.mode == 'test':
                self.test()
            else:
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training function, with per-epoch model saving
        """

        for epoch in range(self.current_epoch, self.config.max_epoch):
            print("[Epoch-{}/{}]".format(epoch+1, self.config.max_epoch))
            self.current_epoch = epoch
            self.scheduler.step(epoch)
            self.train_one_epoch()
            print("\nValidating...")
            valid_acc, valid_loss = self.validate()
            self.scheduler.step(valid_loss)
            print("="*80)

            is_best = valid_acc > self.best_valid_acc
            if is_best:
                self.best_valid_acc = valid_acc

            self.save_checkpoint(is_best=is_best)

    def train_one_epoch(self):
        """
        One epoch training function
        """
        # Initialize tqdm
        if self.config.in_notebook:
            tqdm_batch = tqdm.notebook.tqdm(enumerate(self.data_loader.train_loader), 
                                            total=self.data_loader.train_iterations,
                                            desc="Training phase ") #.format(self.current_epoch+1, self.config.max_epoch))
        else:
            tqdm_batch = tqdm(enumerate(self.data_loader.train_loader), 
                              total=self.data_loader.train_iterations,
                              desc="Training phase ") #.format(self.current_epoch+1, self.config.max_epoch))

        # Set the model to be in training mode (for batchnorm)
        self.model.train()
        # Initialize your average meters
        epoch_loss, epoch_acc = [], []

        for idx, sampled_batch in tqdm_batch:
            if idx == 2 :
                break
#             if self.cuda:
#                 x, y = x.pin_memory().cuda(async=self.config.async_loading), y.cuda(async=self.config.async_loading)
            x, y = sampled_batch['features'], sampled_batch['label']
            # model
            pred = self.model(x)
            # loss
            cur_loss = self.loss(pred, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            # optimizer
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()

            ## append iteration loss
            epoch_loss.append(cur_loss.item())
            
            ## append iteration accuracy
            _, y_pred = torch.max(F.softmax(pred,dim=1), 1)
            correct = (y_pred==y).sum().item()
            running_acc = correct/y.size(0)
            epoch_acc.append(running_acc)
            
            self.current_iteration += 1
            # exit(0)
            sys.stdout.write("\r     train_loss = %.5f, train_acc = %.5f" 
                         % (np.mean(epoch_loss), np.mean(epoch_acc)))
            sys.stdout.flush()
            
        self.summary_writer.add_scalar("epoch_training/loss", np.mean(epoch_loss), self.current_iteration)
        self.summary_writer.add_scalar("epoch_training/acc", np.mean(epoch_acc), self.current_iteration)
        
        tqdm_batch.close()

#         print("[Training Results at epoch-" + str(self.current_epoch) + " ] " 
#               + "loss: " + str(np.mean(epoch_loss)) 
#               + " - acc-: " + str(np.mean(epoch_acc)))

    def validate(self):
        """
        One epoch validation
        :return:
        """
        if self.config.in_notebook:
            tqdm_batch = tqdm.notebook.tqdm(enumerate(self.data_loader.valid_loader), 
                                            total=self.data_loader.valid_iterations,
                                            desc="Valiation phase ") #.format(self.current_epoch+1, self.config.max_epoch))
        else:
            tqdm_batch = tqdm(enumerate(self.data_loader.valid_loader), 
                              total=self.data_loader.valid_iterations,
                              desc="Valiation phase ") #.format(self.current_epoch+1, self.config.max_epoch))

        # set the model in training mode
        self.model.eval()

        epoch_loss, epoch_acc = [], []

        for idx, sampled_batch in tqdm_batch:
            if idx==2:
                break
#             if self.cuda:
#                 x, y = x.pin_memory().cuda(async=self.config.async_loading), y.cuda(async=self.config.async_loading)
            x, y = sampled_batch['features'], sampled_batch['label']
            # model
            pred = self.model(x)
            # loss
            cur_loss = self.loss(pred, y)

            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during Validation.')

            ## append iteration loss
            epoch_loss.append(cur_loss.item())

            ## append iteration accuracy
            _, y_pred = torch.max(F.softmax(pred,dim=1), 1)
            correct = (y_pred==y).sum().item()
            running_acc = correct/y.size(0)
            epoch_acc.append(running_acc)
            
            self.current_iteration += 1
            # exit(0)
            sys.stdout.write("\r    val_loss = %.5f, val_acc = %.5f" 
                         % (np.mean(epoch_loss), np.mean(epoch_acc)))
            sys.stdout.flush()
        
        self.summary_writer.add_scalar("epoch_validation/loss", np.mean(epoch_loss), self.current_iteration)
        self.summary_writer.add_scalar("epoch_validation/acc", np.mean(epoch_acc), self.current_iteration)
        tqdm_batch.close()
        
#         print("[Validation Results at epoch-" + str(self.current_epoch) + " ] " 
#             + "loss: " + str(np.mean(epoch_loss)) 
#             + " - acc-: " + str(np.mean(epoch_acc)))

        return np.mean(epoch_acc), np.mean(epoch_loss)

    def test(self):
        # TODO
        pass

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
#         self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.data_loader.finalize()
        
        
        