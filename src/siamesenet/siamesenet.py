import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from src.siamesenet.cost import *
from src.siamesenet.layers import *
from src.utils import *
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
import numpy

class SiameseNetwork:
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.cfg = cfg
        self.base_network = create_base_network(input_shape)

        self.input_a = Input(shape=input_shape)
        self.input_b = Input(shape=input_shape)
        # processed_a = siamese_base_net(input_a)
        # processed_b = siamese_base_net(input_b)
        self.processed_a = self.base_network(self.input_a)
        self.processed_b = self.base_network(self.input_b)
        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([self.processed_a, self.processed_b])
        self.model = Model([self.input_a, self.input_b], distance)
        # keras.utils.plot_model(model, "siamModel.png", show_shapes=True)
        self.model.summary()

    def train(self, tr_pairs, tr_y, te_pairs, te_y, epochs):
        if self.cfg['have_siamese_model']:
            self.model = tf.keras.models.load_model('./models/siamese_model.h5')
        else:
            rms = RMSprop()
            self.model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
            # p0 = tr_pairs[:, 0]
            # p1 = tr_pairs[:, 1]
            history = self.model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                                batch_size=self.cfg['batch_size'],
                                epochs=epochs, verbose=2,
                                validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
            self.model.save('./models/siamese_model.h5')
        return self.model

    def predict(self, x, batch_sizes):
        # compute the siamese embeddings of the input data
        batch_size = min(len(x), batch_sizes)
        batches = make_batches(len(x), batch_size)
        y_preds = np.array([])
        # iii = self.model.input[0]
        # ooo = self.model.get_layer('model').output
        # predict_model = Model(self.model.input, self.model.get_layer('model').output)
        # self.base_network.predict()
        for i, (batch_start, batch_end) in enumerate(batches):
            y_pred = self.base_network.predict(x[batch_start:batch_end], batch_size=batch_size)
            if(i==0):
                y_preds = y_pred
            else:
                y_preds = np.append(y_preds, y_pred, axis=0)
        return y_preds