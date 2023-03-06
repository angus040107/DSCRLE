import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

class Orthonorm(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        # self.name = name

    def call(self, inputs, epsilon=1e-6):
        x = inputs
        x_2 = tf.matmul(tf.transpose(x), x)
        x_2 += tf.eye(x_2.get_shape()[0]) * epsilon
        L = tf.linalg.cholesky(x_2)
        ortho_weights = tf.transpose(tf.linalg.inv(L)) * tf.sqrt(tf.cast(self.units, dtype=tf.float32))

        return tf.matmul(x, ortho_weights)

def make_layer_list(arch, network_type=None, reg=None, dropout=0):
    '''
    Generates the list of layers specified by arch, to be stacked
    by stack_layers (defined in src/core/layer.py)

    arch:           list of dicts, where each dict contains the arguments
                    to the corresponding layer function in stack_layers

    network_type:   siamese or spectral net. used only to name layers

    reg:            L2 regularization (if any)
    dropout:        dropout (if any)

    returns:        appropriately formatted stack_layers dictionary
    '''
    layers = []
    for i, a in enumerate(arch):
        layer = {'l2_reg': reg}
        layer.update(a)
        if network_type:
            layer['name'] = '{}_{}'.format(network_type, i)
        layers.append(layer)
        if a['type'] != 'Flatten' and dropout != 0:
            dropout_layer = {
                'type': 'Dropout',
                'rate': dropout,
                }
            if network_type:
                dropout_layer['name'] = '{}_dropout_{}'.format(network_type, i)
            layers.append(dropout_layer)
    return layers


def stack_layers(layers):
    slayers = []
    for layer in layers:
        # check for l2_reg argument
        l2_reg = layer.get('l2_reg')
        if l2_reg:
            l2_reg = l2(layer['l2_reg'])

        # create the layer
        if layer['type'] == 'Dense':
            l = Dense(layer['size'], activation=layer.get('activation'), kernel_regularizer=l2_reg)
        elif layer['type'] == 'Conv2D':
            l = Conv2D(layer['channels'], kernel_size=layer['kernel'], activation='relu', kernel_regularizer=l2_reg)
        elif layer['type'] == 'BatchNormalization':
            l = BatchNormalization()
        elif layer['type'] == 'MaxPooling2D':
            l = MaxPooling2D(pool_size=layer['pool_size'])
        elif layer['type'] == 'Dropout':
            l = Dropout(layer['rate'])
        elif layer['type'] == 'Flatten':
            l = Flatten()
        elif layer['type'] == 'Orthonorm':
            l = Orthonorm(layer.get('batchsize'));
        else:
            raise ValueError("Invalid layer type '{}'".format(layer['type']))
        slayers.append(l)

    return slayers
