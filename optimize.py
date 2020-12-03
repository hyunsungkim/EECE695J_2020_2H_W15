#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
from model import myModel
import os
import copy


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.random.set_seed(0)
BATCH_SIZE = 16


# In[ ]:


def plot_weight_dist(w_dict, b_dict):
    """
        This function plots distribution of parameters. The
        second subplot describes the histogram of the absolute
        values.
    """
    w = []
    for item in w_dict.values():
        w.extend(item.ravel())

    for item in b_dict.values():
        w.extend(item.ravel())
    
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(5,5))
    ax1.set_title('Weight distribution')
    ax1.hist(np.array(w).flatten(), bins=100)
    ax2.hist(np.array(np.abs(w)).flatten(), bins=100)

# Load the testing dataset and the construct a dataloader
data = np.load('./data/testset.npz')
test_x, test_y = data['test_x'], data['test_y']
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_dataset = test_dataset.shuffle(buffer_size=10000).batch(BATCH_SIZE)

# Load the pre-trained model
model = myModel()
baseline = tf.train.latest_checkpoint('./models/')
model.load_weights(baseline).expect_partial()
opt = tf.keras.optimizers.Adam(learning_rate = 1e-3)
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
model.build(input_shape=(BATCH_SIZE,10))
#model.summary()

# Evaluate the baseline model
print("\nTesting the baseline model")
loss_base, accuracy_base = model.evaluate(test_dataset)

# Load parameters into dictionaries, which will be modified by you
w_dict, b_dict = model.get_params()
plot_weight_dist(w_dict, b_dict) 


# In[ ]:


w_dict_new = copy.deepcopy(w_dict)
b_dict_new = copy.deepcopy(b_dict)

############## BEGGINING of the student code. ##################
# Write your codes FROM HERE to manipulate model parameters in
# two dictionaries, i.e., w_dict_new and b_dict_new, using numpy
# operations. One example is to apply hard-thresholding on small
# weight parameters with a fixed threshold as follows.
#   
# w_dict["fc1"][np.abs(w_dict["fc1"])<1e-3] = 0
# b_dict["fc1"][np.abs(b_dict["fc1"])<1e-3] = 0
#
# For the basic indexing of the numpy array, refer to followings
# https://numpy.org/doc/stable/user/basics.indexing.html#
# https://cs231n.github.io/python-numpy-tutorial/#array-indexing

# Enter your name and ID number
name = ""
id = 0

# Prune network parameters (i.e. make some parameters to zeros)
w_dict_new["fc1"][np.abs(w_dict_new["fc1"])<1e-1] = 0 # Example
b_dict_new["fc1"][np.abs(b_dict_new["fc1"])<1e-1] = 0 # Example

# Enter bit-widths of integer and fraction part of resultant
# fixed point numbers. Enter `None` if you don't want to apply
# quantization
intbit = None
fracbit = None

##################### END of the student code. ################

assert (len(name)!="") and (id>0), "Invalid name or id. Please enter your name and id"

# Plot the parameter distribution
plot_weight_dist(w_dict_new, b_dict_new)

# Reload the model
my_model = myModel()
baseline = tf.train.latest_checkpoint('./models/')
my_model.load_weights(baseline).expect_partial()
opt = tf.keras.optimizers.Adam(learning_rate = 1e-3)
my_model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
my_model.build(input_shape=(BATCH_SIZE,10))

# Optimize the model
my_model.update_params(w_dict_new, b_dict_new)
my_model.quantize_params(intbit=intbit, fracbit=fracbit)

# Test the optimized model
print("Testing your model")
size, size_base, drop_ratio = my_model.compute_model_size()
loss, accuracy = my_model.evaluate(test_dataset)

# Print the result
print("\n"+"="*15+"REPORT THIS RESULT"+"="*15)
print(f"{name}, {id}")
print()
print(f"MODEL\t\tSIZE(B)\tACCURACY")
print(f"Original model\t {size_base:4d}\t {accuracy_base*100:.2f}%")
print(f"Your model\t {size:4d}\t {accuracy*100:.2f}%")
print()
if(my_model.quantized):
    print(f"Quantization: \t({my_model.intbit+my_model.fracbit},{my_model.intbit},{my_model.fracbit})")
else:
    print(f"Quantization: \tNo quantization")
print(f"Size reduction: {(1-size/size_base)*100:3.2f}%")
print(f"Accuracy drop: \t{(accuracy_base-accuracy)*100:.2f}%")
print("="*49)

model.save_weights(f"./models/student/model_{name}_{id}.ckpt")


# In[ ]:




