import numpy as np

from src.dscrle.cost import *
from src.dscrle.layer import *
from tensorflow.keras.models import Model

class mymodel(Model):
    def __init__(self, cfg):
        super().__init__()
        self.layerlists = make_layer_list(cfg['arch'], 'spectral', cfg['spec_reg'])
        self.layerlists += [{'type': 'Dense', 'activation': 'tanh', 'size': cfg['n_clusters']}]
        self.layerlists += [{'type': 'Orthonorm', 'name':'orthonorm', 'batchsize': cfg['batch_size']}]
        self.slayers = stack_layers(self.layerlists)
        self.data_dim = cfg['data_dim']


    def call(self, inputs):
        # output = input

        for slayer in self.slayers:
            inputs = slayer(inputs)

        return inputs

    def calloss(self, X, y_pred, cfg):
        # mscale = get_scale( X.reshape( (cfg['batch_size'], -1) ), cfg['batch_size'] )
        # AffinityMatrix = full_affinity(tf.reshape(X, [cfg['batch_size'], -1]), scale = mscale)
        if(cfg['affinity'] == 'anchor'):
            AffinityMatrix = anchor_affinity(tf.reshape(X, [cfg['batch_size'], -1]), n_nbrs = cfg['n_nbrs'])
            DYY = squared_distance(y_pred)
            loss = tf.reduce_sum(AffinityMatrix * DYY) / tf.cast(cfg['batch_size'], dtype=tf.float32)

        elif (cfg['affinity'] == 'full'):
            AffinityMatrix = full_affinity(tf.reshape(X, [cfg['batch_size'], -1]))
            DYY = squared_distance(y_pred)
            loss = tf.reduce_sum(AffinityMatrix * DYY) / tf.cast(cfg['batch_size'], dtype=tf.float32)

        elif (cfg['affinity'] == 'knn'):
            AffinityMatrix = knn_affinity(tf.reshape(X, [cfg['batch_size'], -1]), n_nbrs=cfg['n_nbrs'])
            DYY = squared_distance(y_pred)
            loss = tf.reduce_sum(AffinityMatrix * DYY) / tf.cast(cfg['batch_size'], dtype=tf.float32)

        elif (cfg['affinity'] == 'sec'):
            self.gammad = 0.5
            self.miu = 1e-5
            self.Wxbf = tf.Variable(tf.random.uniform([self.data_dim, cfg['n_clusters']], name='wxbf'), trainable=False)
            [AffinityMatrix, self.Wxbf] = sec(X, y_pred, cfg['n_nbrs'], self.data_dim, self.gammad, self.Wxbf)
            DYY = squared_distance(y_pred)
            sc_loss = tf.reduce_sum(AffinityMatrix * DYY) / tf.cast(cfg['batch_size'], dtype=tf.float32)
            secloss = tf.reduce_sum(tf.norm(tf.matmul(X, self.Wxbf) - y_pred, axis=-1)) / tf.cast(cfg['batch_size'], dtype=tf.float32)
            ct_loss = tf.reduce_sum( tf.norm( self.Wxbf, axis=-1 ) ) / tf.cast(cfg['batch_size'], dtype=tf.float32)
            loss = sc_loss + self.miu*self.gammad*secloss + self.miu*(1.0-self.gammad)*ct_loss
        elif (cfg['affinity'] == 'kec'):
            self.gammad = 0.5
            self.miu = 1e-5
            self.Wxbf = tf.Variable(tf.random.uniform([self.data_dim, cfg['n_clusters']], name='wxbf'), trainable=False)
            [AffinityMatrix, self.Wxbf] = kec(X, y_pred, cfg['n_nbrs'], self.data_dim, self.gammad, self.Wxbf)
            DYY = squared_distance(y_pred)
            sc_loss = tf.reduce_sum(AffinityMatrix * DYY) / tf.cast(cfg['batch_size'], dtype=tf.float32)
            secloss = tf.reduce_sum(tf.norm(tf.matmul(X, self.Wxbf) - y_pred, axis=-1)) / tf.cast(cfg['batch_size'], dtype=tf.float32)
            ct_loss = tf.reduce_sum( tf.norm( self.Wxbf, axis=-1 ) ) / tf.cast(cfg['batch_size'], dtype=tf.float32)
            loss = sc_loss + self.miu*self.gammad*secloss + self.miu*(1.0-self.gammad)*ct_loss
        return loss

class SpectralNet():
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = mymodel(self.cfg)

    def train(self, data_loader):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg['spec_lr'])
        num_batches = int(data_loader.num_train_data // self.cfg['batch_size'] * self.cfg['spec_ne'])
        for batch_index in range(num_batches):
            X, _ = data_loader.get_batch(self.cfg['batch_size'])
            with tf.GradientTape() as tape:
                y_pred = self.model(X)
                # III = tf.cast( tf.matmul( tf.transpose(y_pred), y_pred), dtype=tf.uint8 )
                loss = self.model.calloss(X, y_pred, self.cfg)
            grads = tape.gradient(loss, self.model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, self.model.variables))
            print("batch %d: loss %f" % (batch_index, loss))

        return self.model

    def predict(self, data_loader):
        num_batches = (data_loader.num_test_data + self.cfg['batch_size'] - 1) // self.cfg['batch_size']
        batches = [(i*self.cfg['batch_size'], min(data_loader.num_test_data, (i+1)*self.cfg['batch_size'])) for i in range(num_batches)]
        for i, (batch_start, batch_end) in enumerate(batches):
            X = data_loader.test_data[batch_start:batch_end, :]
            if(i==0):
                y_preds = np.array( self.model(X) )
            else:
                y_preds = np.concatenate((y_preds, np.array( self.model(X) )), axis=0)
            # III = tf.cast( tf.matmul( tf.transpose(y_pred), y_pred), dtype=tf.uint8 )
        return y_preds