import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import copy

class myModel(tf.keras.Model):
    def __init__(self):
        super(myModel, self).__init__(name='model')
        self.quantized = False
        self.intbit = None
        self.fracbit = None
        
        self.classifier = [
            layers.InputLayer(input_shape=(10,), name='input'),
            layers.Flatten(name='flatten1'),
            layers.Dense(40, use_bias=True, activation='relu', name='fc1'),
            layers.Dropout(rate = 0.1, name='dropout1'),
            layers.Dense(20, use_bias=True, activation='relu', name='fc2'),
            layers.Dropout(rate = 0.1, name='dropout2'),
            layers.Dense(10, use_bias=True, activation='relu', name='fc3'),
            layers.Dense(2, use_bias=True, activation='relu', name='fc4'),
            layers.Softmax(name='output')
        ]
    
    def call(self, x):
        for layer in self.classifier:
            x = layer(x)
        return x
    
    def get_params(self):
        w_dict = {}
        b_dict = {}

        for layer in self.classifier:
            if("fc" in layer.name):
                w_dict[layer.name] = layer.get_weights()[0]
                b_dict[layer.name] = layer.get_weights()[1]
        return w_dict, b_dict
    
    def update_params(self, w_dict, b_dict):
        for layer in self.classifier:
            if("fc" in layer.name):
                layer.set_weights((w_dict[layer.name], b_dict[layer.name]))
                
    def quantize_params(self, intbit, fracbit):
#        if(self.quantized == False):
#            self.w_backup, self.b_backup = self.get_params()
#        if(self.intbit != intbit or self.fracbit != fracbit):
#            assert self.quantized==False, "Repeated quantization operation. Load the model again to apply other quantization scheme"
        if(intbit == None and fracbit == None):
            print("Skipping quantization")
            return
        assert ((intbit+fracbit) % 8 == 0) and (intbit+fracbit <= 32), "Invalid bitwidth, only 8-bit, 16-bit, and 32-bit are allowed"
        self.intbit = intbit
        self.fracbit = fracbit
        for layer in self.classifier:
            if("fc" in layer.name):
                [w, b] = layer.get_weights()
                w = self._quantize(w, self.intbit, self.fracbit)
                b = self._quantize(b, self.intbit, self.fracbit)
                layer.set_weights((w, b))
        self.quantized = True
    
    def compute_model_size(self):
        w = []
        for item in self.trainable_variables:
            w.extend(item.numpy().ravel())
        w = np.array(w)
        
        size_base = len(w)*4
        drop_ratio = (len(w)-np.sum(w!=0))/len(w)
        if(self.quantized):
            size = np.sum(w!=0)*(self.intbit+self.fracbit)*(2**(-3))
        else:
            size = np.sum(w!=0)*4
        return size.astype(int), size_base, drop_ratio
    
    def _quantize(self, input, intbit, fracbit):
        input = np.around(input*(2**fracbit))/(2**fracbit)
        input = np.clip(input, -(2**intbit), 2**intbit-2**(-fracbit))
        return input
        
    