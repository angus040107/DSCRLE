import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from collections import defaultdict

def returncfg():
    # PARSE ARGUMENTS
    params = defaultdict(lambda: None)

    # SET GENERAL HYPERPARAMETERSd
    general_params = {
        'dset': 'Botswana',
        'vae_params_path': 'src/vae/configs/vae.yaml',
        'spatial': 5,
        'affinity': 'sec',
        'n_nbrs': 3,  # numb
        # er of nonzero entries (neighbors) to use for graph Laplacian affinity matrix
        'spec_ne': 500,  # number of training epochs for spectral net
        'spec_lr': 1e-4,  # initial learning rate for spectral net

        'siamese_tot_pairs': 200000,  # total number of pairs for siamese net
        'siam_k': 2,
        'siam_ne': 100,  # number of training epochs for siamese net
        'batch_size': 1024,  # batch size fo r spectral net
        'siam_lr': 1e-4,  # initial learning rate for siamese net

        'arch': [  # network architecture. if different architectures are desired for siamese net and
            #   spectral net, 'siam_arch' and 'spec_arch' keys can be used
            {'type': 'Dense', 'activation': 'relu', 'size': 500},
            # {'type': 'BatchNormalization'},
            # {'type': 'Dropout', 'rate': 0.1},
            {'type': 'Dense', 'activation': 'relu', 'size': 500},
            # {'type': 'BatchNormalization'},
            # {'type': 'Dropout', 'rate': 0.1},
            {'type': 'Dense', 'activation': 'relu', 'size': 200},
            # {'type': 'BatchNormalization'},
            # {'type': 'Dropout', 'rate': 0.1},
            {'type': 'Dense', 'activation': 'relu', 'size': 100},
            # {'type': 'BatchNormalization'},
            # {'type': 'Dropout', 'rate': 0.1},
            {'type': 'Dense', 'activation': 'tanh', 'size': 50},
        ],
    }
    params.update(general_params)
    return params

# cfg = returncfg()
# print(cfg)