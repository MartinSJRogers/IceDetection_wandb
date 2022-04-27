# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:13:37 2022

@author: marrog
simple unet to check if wandb works
"""


from src.model.uNet_ifElse import uNet
from src.utils import dataParser
from src.utils.dataParser import DataParser

import numpy as np

import argparse

import os


from keras.utils.vis_utils import plot_model
from keras import backend as K
from tensorflow.keras import callbacks as callbacks

import wandb
from wandb.keras import WandbCallback
import subprocess
subprocess.call("dir", shell=True)
#from callbacks import BatchwiseWandbLogger

####Argparse commands. To do################
parser = argparse.ArgumentParser()

#parser.add_argument('--seed', default=42, type=int)
parser.add_argument("--wandb", help="Use Weights and Biases",
                    default=False, action="store_true")
parser.add_argument('--learning_rate', default=0.0005, type=float)
parser.add_argument('--batch_size', default=2, type=int)
#parser.add_argument('--n_filters_factor', default=2, type=float)
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--n_layers', default=4, type=float)
parser.add_argument('--start_neurons_number', default=64, type=int)

args = parser.parse_args()

#seed = args.seed
####################################################################

#### Set up wandb
####################################################################
wandb.init(project="my-test-project", entity="martinrogers")


wandb_mode='online'
n_classes=4# number of classes referenced layer classifies image into
class_weights=np.array([0.1,0.1,0.9,0.9])

##To use when argparse is set up ####
"""
defaults= dict(epochs = args.epochs, 
               batch_size = args.batch_size, 
               learning_rate=args.learning_rate)
"""
defaults= dict(epochs = 1, 
               batch_size = 4, 
               learning_rate=0.001,
               n_layers=4, 
               start_neurons_number=64)

wandb.init(
    project='martin-test-Feb17',
    entity='martinrogers',
    config=defaults,
    allow_val_change=True,
    mode=wandb_mode,
)


print(wandb.config)

# Whether to run the custom callbacks at the 0th batch
sample_callbacks_at_zero = False
mcMonitor = 'val_acc_mean'
mcMode = 'max'
mcMonitor2 = 'val_loss'
mcMode2 = 'min'

def generate_minibatches(data_parser, train=True):
    #pdb.set_trace()
    while True:
        if train:
            batch_ids = np.random.choice(dataParser.training_ids, dataParser.batch_size_train)
        else:
            batch_ids = np.random.choice(dataParser.validation_ids, dataParser.batch_size_train*2)
        ims, ems, _ = dataParser.get_batch(batch_ids)
        print(batch_ids)
        #show(ems[0:1,:, :])
        yield(ims, ems)

######
if __name__ == "__main__":
    # params- create folder called 'HEDSeg', adds csv and model folder in there
    #checkpoint_fn: the model checkpoints will be saved with the epoch number and the validation loss in the filename
    model_name = 'UNetModelTrials'
    model_dir     = os.path.join('introCheckpoints', model_name)
    csv_fn        = os.path.join(model_dir, 'intro_trainLog.csv')
    checkpoint_fn = os.path.join(model_dir, 'intro_weights-advancementsDefault-{epoch:02d}.hdf5')#try different name

    #checkpoint_fn = os.path.join(model_dir, 'intro_weights-advancementsDefault-{epoch:02d}-{val_loss:.2f}.hdf5')#try different name

    
    

    # environment
    K.set_image_data_format('channels_last')#(h,w,band)
    K.image_data_format()
    


    #setting to -1 ensures the CPU is used. Current issue: when using GPU 
    #memory limit is 2GB so get OMM out of memory error. 
    #May need to use HPC when using larger image databases. 
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"
    print('DEVICES', os.environ["CUDA_VISIBLE_DEVICES"])
    from tensorflow.python.client import device_lib
    #print( device_lib.list_local_devices())
    if not os.path.isdir(model_dir): os.makedirs(model_dir)

    # prepare data
    #dataparser holds original RGB image, corresponding binary shoreline image
    #and file names. For every image.
    dataParser = DataParser(wandb.config.batch_size)

    # model- make pdf of it. 
    start_neurons_number=64
    model = uNet(n_classes, wandb.config.start_neurons_number, 
                 wandb.config.learning_rate, wandb.config.n_layers)
    #model = uNet()
    #plot_model(model, to_file=os.path.join(model_dir, 'introModel.pdf'), show_shapes=True)

    # training
    # call backs. 
    #MOdelCheckpoint- save infomration at end of every epoch. versbose=1 shows progress bar [------]
    #csv_logger- save info to csv
    #tensorboard- save info as Tensorboard object. 
    checkpointer = callbacks.ModelCheckpoint(filepath=checkpoint_fn, verbose=1, monitor='val_loss',save_best_only=True)
    csv_logger  = callbacks.CSVLogger(csv_fn, append=True, separator=';')#write info in csv file
    tensorboard = callbacks.TensorBoard(log_dir=model_dir, histogram_freq=1, 
                                        write_graph=False, write_grads=False, write_images=False)

    wandb_callbacks = []
    
    
    wandb_callbacks.append(
        WandbCallback(
            monitor=mcMonitor2, mode=mcMode2,
            log_weights=False, log_gradients=False
        )
    )
    

    train_history = model.fit(
                        generate_minibatches(dataParser,),# for training data
                        #max_q_size=40, workers=5,
                        steps_per_epoch=1,#dataParser.steps_per_epoch,  ## docs say this should be equal to #samples/ training_batch_size
                        epochs=wandb.config.epochs,# not 2048*2
                        #class_weight= class_weights,
                        validation_data=generate_minibatches(dataParser, train=False), # for validation data
                        validation_steps=dataParser.validation_steps,##validation ids /batch_size_train(10)*2
                        callbacks=[checkpointer, csv_logger, tensorboard, wandb_callbacks])

    
    
    """
    wandb_callbacks.append(
        BatchwiseWandbLogger(
            batch_frequency=wandb.config.epochs,
            log_weights=True,
            sample_at_zero=sample_callbacks_at_zero
        )
    )
    for i in range(wandb.config.epochs):
        wandb.log({"acc'": train_history.history['val_loss'][i]})
        wandb.log({"acc'": train_history.history['accuracy'][i], 'val_acc': train_history.history['val_accuracy'][i],
                   "loss": train_history.history['loss'][i], "val_loss": train_history.history['val_loss'][i]})
    
    wandb.log({'epoch': wandb.config.epochs, 'acc': train_history.history['accuracy']})
    
    print(train_history.history['loss'][0])
    for key in train_history.history:
        print(key)
    #pdb.set_trace()

    model_json=model.to_json()
    json_fp=os.path.join(model_dir,"introModel.json")
    with open (json_fp, "w") as json_file:
        json_file.write(model_json)
    weights_fn = os.path.join(model_dir, 'intoWeights20.hdf5')

    model.save_weights(weights_fn)
    print("saved model to disk")
    """


