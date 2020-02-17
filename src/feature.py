import numpy as np

import os
import sys
import shutil

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


import librosa 
import wave

import torch 
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

from torchvision import transforms, utils

class AudioDataLoader:
    def __init__(self, config):
        self.config = config
        assert self.config.mode in ['train', 'test']
                
        if self.config.mode == 'train':
          
            # ===============================
            ## TO DO : Check again the transform (Normalize, Logmel Spectro,etc... better with or without librosa)
            self.transform = transforms.Compose([Random_Sample(sample_duration=self.config.sample_duration,
                                                               sampling_rate=self.config.sampling_rate),
                                                 LogMelSpectrogram(sampling_rate = self.config.sampling_rate)])
            # ===============================
                                                
            transformed_audio_ds = AudioFeaturesExtractor(csv_filename = self.config.csv_filename,
                                                         dataset_path = self.config.dataset_path,
                                                         mode = self.config.mode,
                                                         sampling_rate = self.config.sampling_rate,
                                                         transform = self.transform)
          
            self.classes = transformed_audio_ds.get_classes()
          
            split_rate = self.config.train_valid_split_rate
            
            dataset_size = len(transformed_audio_ds)
            train_size = int(split_rate * dataset_size)
            valid_size = dataset_size - train_size
            train_set, valid_set = random_split(transformed_audio_ds, [train_size, valid_size])
            
            self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False)
            
            self.train_iterations = len(self.train_loader)
            self.valid_iterations = len(self.valid_loader)
            
        elif self.config.mode == 'test':
            print("TO DO")
        
        else:
            raise Exception("Please choose a mode for data loading. Optional modes : ['train', 'test']")
       
    def finalize(self):
        pass
        
class AudioFeaturesExtractor(Dataset):
    
    def __init__(self, csv_filename, dataset_path, mode, sampling_rate=16000, transform=None):
        csv_filepath = os.path.join(dataset_path, csv_filename)
        self.data = pd.read_csv(csv_filepath)
        self.mode = mode
        if self.mode == 'train':
            self.root_dir = os.path.join(dataset_path, 'audio_train/')
        elif self.mode == 'test':
            self.root_dir = os.path.join(dataset_path, 'audio_test/')
        self.transform = transform
        self.sampling_rate = sampling_rate
        
        
        print("Nb of training samples = ", self.__len__())

        if self.mode == 'train' :
            self.all_classes = list(self.data['label'].unique())
            self.n_classes = len(self.all_classes)
            print('Nb of classes = ', self.n_classes)

    ## len(dataset) return the size of dataset    
    def __len__(self):
        return self.data.shape[0]
    
    def get_classes(self):
        assert self.mode == 'train'
        return self.all_classes
    
    ## dataset[i] return the ith sample
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        fname = os.path.join(self.root_dir,
                            self.data.iloc[idx,0])
        
        wav, sr = torchaudio.load(fname, normalization=True)  ## wav is a tensor, not an array
        ## Resampling with new sampling_rate
        sample = torchaudio.transforms.Resample(sr, self.sampling_rate)(wav)
#         sample = wav.t()
        
        ## Random offset + Padding + Normalization
        if self.transform :
#             sample_wave = self.transform[0](sample)
            features = self.transform(sample)

            if self.mode == 'train':
                label_name = self.data.iloc[idx,1]
                label = self.all_classes.index(label_name)
                return {'features':features, 'label':label}

            elif self.mode == 'test' : 
                return {'features':features}
        
        else:
            return {'raw_signal':sample}
    

class Random_Sample(object):
    
    def __init__(self, sample_duration, sampling_rate):
        self.sample_length = sample_duration * sampling_rate

    def __call__(self, sample):
        original_audio_length = sample.size(1)
        sample = sample.t()
        ## random offset / Padding
        if original_audio_length > self.sample_length:
            max_offset = original_audio_length - self.sample_length
            offset = np.random.randint(max_offset)
            new_sample = sample[offset:(offset+self.sample_length)].numpy()
        else:
            if original_audio_length == self.sample_length:
                offset = 0
            else: 
                max_offset = self.sample_length - original_audio_length
                offset = np.random.randint(max_offset)
            ## zero padding 
            new_sample = np.pad(sample.numpy()[:,0], (offset, self.sample_length - original_audio_length - offset), "constant")
                
        ## Normalize
        max_val = np.max(new_sample)
        min_val = np.min(new_sample)
    
        new_sample = (new_sample - min_val)/(max_val - min_val + 1e-6)
        new_sample -= 0.5
        
        return torch.from_numpy(new_sample.reshape(1,-1))
    
class MFCC_Features(object):
    def __init__(self,sampling_rate):
        self.sampling_rate = sampling_rate
    def __call__(self,sample):
        mfcc = torchaudio.compliance.kaldi.mfcc(sample,sample_frequency=self.sampling_rate)
        return mfcc

class Spectrogram(object):
    def __init__(self,sampling_rate):
        self.sampling_rate = sampling_rate
    def __call__(self,sample):
        spectrogram = torchaudio.transforms.Spectrogram()(sample)
        spectrogram[0,:,:] = spectrogram.log2()[0,:,:]
        return spectrogram
    
class LogMelSpectrogram(object):
    def __init__(self,sampling_rate):
        self.sampling_rate = sampling_rate
    def __call__(self,sample):
        mel_spectrogram = torchaudio.transforms.MelSpectrogram()(sample)
        mel_spectrogram[0,:,:] = mel_spectrogram.log2()[0,:,:]
        mel_spectrogram[mel_spectrogram == -float('inf')] = 0.0
        return mel_spectrogram