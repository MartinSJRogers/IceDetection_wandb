# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:18:16 2021

@author: marrog
"""
import os
import numpy as np


#go through file list (imageNames.lst). 
#read each line, remove spaces
def read_file_list(filelist):
    
    pfile = open(filelist)# os module e.g. os.path.strip
    filenames = pfile.readlines()
    pfile.close()
    filenames = [f.strip() for f in filenames]#remove new line symbol
    return filenames
#split into raw and binary images
def split_pair_names(filenames, base_dir):
    filenames = [c.split('\t') for c in filenames] #columns split by tab
    filenames = [(os.path.join(base_dir, c[0]), os.path.join(base_dir, c[1])) for c in filenames]
    return filenames


class DataParser():
    #constructuor- sets lots of defaults. sets an instance of class. what should i set the values of variables instance.
    #sensible defaults are set in the instance of the class. 
    #e.g. the class should have an image width. If undefined forces it to have a sensible value (here 480).
    #self- refers to this class. 
    def __init__(self, batch_size_train):
        self.train_file=r"C:/Users/marrog/OneDrive - NERC/Documents/iceDetection/detection/images/preprocess/subsetEdges/fileListSmall.txt"
        self.train_data_dir = r'C:/Users/marrog/OneDrive - NERC/Documents/iceDetection/detection/images/preprocess/subsetEdges'
        self.training_pairs = read_file_list(self.train_file)# training pairs assumed to be raw and binary image
        print('Number of training pairs',len(self.training_pairs))
        self.samples = split_pair_names(self.training_pairs, self.train_data_dir)# diff array grid for each pair
        
        #find number of sample pairs (len), get range (e.g. 0-4) then random shuffle. 
        #Had to place in a list for range shuffle top work
        self.n_samples = len(self.training_pairs)
        print(self.n_samples)
        self.all_ids = range(self.n_samples)
        np.random.shuffle(list(self.all_ids))

        #Training images become first 0.8*length of shuffled images. Validation= last 0.2
        train_split = 0.8# changes from 0.8
        self.training_ids = self.all_ids[:int(train_split * len(self.training_pairs))]
        self.validation_ids = self.all_ids[int(train_split * len(self.training_pairs)):]

        #divide % training images by batch size train (10). remainder must =0
        #work out steps per epoch training data/10
        self.batch_size_train = batch_size_train
        #assert len(self.training_ids) % batch_size_train == 0 
        self.steps_per_epoch = len(self.training_ids)/batch_size_train
        self.steps_per_epoch=5# remember to remove
        print('steps_per_epoch', self.steps_per_epoch, self.n_samples)
        
        #remainder of validation data/20 must =0. there a multiple of 20. 
        #Means there must be a minimum of 100 samples. 
        #assert len(self.validation_ids) % (batch_size_train*2) == 0
        self.validation_steps = int(np.floor(len(self.validation_ids)/(batch_size_train*2)))
        
        self.image_width = 480
        self.image_height = 480
        self.target_regression = True
        
        #randomly choose training_ids (first 80% of all ids). Pick number of examples
        #equal to batch size (i.e. 2, 4 or 10). Send to get_batch method

    #this is called in generator method to actually get images 
    #corresponding to id numbers
    def get_batch(self, batch):
        
        filenames = []
        images = []
        edgemaps = []
        filedr= r"C:/Users/marrog/OneDrive - NERC/Documents/iceDetection/detection/images/preprocess/subset"
        #print('Batch_Ids', batch)
        for idx, b in enumerate(batch):

            raster=np.load(self.samples[b][0])
            

            raster=np.swapaxes(raster, 0, 2)
            raster=np.swapaxes(raster, 0, 1)

                      
            #.npy file
            em=np.load(self.samples[b][1])

            
            #em = np.array(em.convert('L'), dtype=np.float32)
            
            if self.target_regression:
                bin_em = em / 255.0
            else:
                bin_em = np.zeros_like(em)
                bin_em[np.where(em)] = 1
            

            bin_em = np.expand_dims(bin_em, 2)
            #print('em shape now', np.shape(bin_em))
            images.append(raster)
            edgemaps.append(bin_em)
            filenames.append(self.samples[b])
            
        images   = np.asarray(images)
        edgemaps = np.asarray(edgemaps)
        
        #return all 3-band images and edgemaps as arrays with associated filenames
        return images, edgemaps, filenames


