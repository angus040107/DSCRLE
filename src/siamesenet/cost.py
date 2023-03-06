from tensorflow.keras import backend as K

def contrastive_loss(y_true, y_pred):

    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(K.cast(y_true, 'float32') * sqaure_pred + K.cast(1 - y_true, 'float32') * margin_square)

def accuracy(y_true, y_pred): # Tensor上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))