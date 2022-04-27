# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
#Some errors:
Found unexpected keys: needed to specify the name of the output layer- simples.
"""

import tensorflow as tf
from keras import backend as K
   
from keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, Dropout, Input, concatenate, BatchNormalization
from tensforflow.keras.models import Model
#from tensorflow import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Input

from tensorflow.keras.optimizers import Adam




def uNet(n_classes, start_neurons, learning_rate):
    
    img_input = Input(shape=(480,480,3), name='input')
    
    a = BatchNormalization()(img_input)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", name='conv1')(a)
    a = BatchNormalization()(conv1)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(a)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #pool1 = Dropout(0.25)(pool1)
    
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)
    
    """
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

   
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    
    """
    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool2)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
    
    """
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    
    """
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    
    
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    
    #for multiclass segmentation, this needs to be equal to the number of classes
    #softmax instead of sigmoid
    output_layer = Conv2D(n_classes, (1,1), padding="same", activation="softmax", name= 'output_layer')(uconv1)
    
    
    # model
    model = Model(inputs=[img_input], outputs=[output_layer])
    #filepath = 'C:/Users/msjr2/Documents/Cambridge/machineLearning/firstPythons/Keras_HED/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    #load_weights_from_hdf5_group_by_name(model, filepath)
    
    #myloss={'output_layer': 'categorical_crossentropy'}
    #model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy())
    model.compile(loss={'output_layer': 'categorical_crossentropy'},
                  optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    
    
    
    #wandb.log({"loss'b": loss, "lossb": 0.1})
    #wandb.log({"loss": 0.2})

    # Optional
    #wandb.watch(model)
    
    return model
    
def loss_function():
    a=1
    return a

    

def load_weights_from_hdf5_group_by_name(model, filepath):
    ''' Name-based weight loading '''

    import h5py

    f = h5py.File(filepath, mode='r')

    flattened_layers = model.layers
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in flattened_layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # we batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            if len(weight_values) != len(symbolic_weights):
                raise Exception('Layer #' + str(k) +
                                ' (named "' + layer.name +
                                '") expects ' +
                                str(len(symbolic_weights)) +
                                ' weight(s), but the saved weights' +
                                ' have ' + str(len(weight_values)) +
                                ' element(s).')
            # set values
            for i in range(len(weight_values)):
                weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
                K.batch_set_value(weight_value_tuples)

#input_layer = Input((img_size_target, img_size_target, 1))
#output_layer = build_model(input_layer, 16)



##Loss function code for HED to help with comparison.

def cross_entropy_balanced(y_true, y_pred):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits
    #simply about converting format of y values to tensors (matrix/ array of any dimension)
    #logit is just the name for the yhat output array at the end of a neural network
    
    #epsilon- 1xe-7. Very small value, y values are subtracted by this value to 
    #prevent any 0 values being divided. 
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    #clip values of y_pred to range between epsilon (1xe-7) and 1 minus epsilon. 
    #values higher and lower than the max and min value become the max and min value respectively
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    #makes 0.5 =0, and values closer to 0 and 1 asumptope to infinity- to do with entrophy?
    y_pred   = tf.math.log(y_pred/ (1 - y_pred))
    
    #converts/ casts to a new data type (float32)
    y_true = tf.cast(y_true, tf.float32)
    
    #count_neg and count_pos are the number of 0s and 1s respectively. 
    #count_pos counts all the pixels with a value of 1 in the entire array
    #count_neg uses (1.0-y_true) to count the number of pixels =0, but uses the 1.0-
    #to acocunt for the fact that all negative pixels =0 so otherwise count would =0.
    #redcue sum reduces the array into the sum of the value of the pixels specified.
    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    # Equation [2]
    #proportion of pixels =0 in image array
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    #inverse- finds proportion of pixels with val =1 in image array
    pos_weight = beta / (1 - beta)
    
    #changed from targets to labels
    #simply working out cost/ difference between true y and predicted y
    #note- y_pred was revalued using log function above. This relates back to entrophy.It means that
    #a greater loss value is determined when a value of 0 occured in y_true, but y_pred has a value of 0.9
    #compared with a value of 0.6.
    #returns 'cost' a tensor with the same shape as y_pred with the loss value per pixel.
    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels. return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def ofuse_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x




