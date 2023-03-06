import sys, os
import time

import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from src.sconfig import returncfg
from src.DEC.DEC import dec_reduction
from src.siamesenet.pairs import create_pairs_from_unlabeled_data
from src.siamesenet.siamesenet import SiameseNetwork
from src.dscrle.network import SpectralNet
from src.utils import ineedtosaveresults, get_results, label2img, imgsave
from src.datatools import get_data, DataLoader
from sklearn.preprocessing import *

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
cfg = returncfg()
ori_input_data, labels, labels_indexs, label_ = get_data(datasetsname=cfg['dset'], cfg=cfg)
# input_data = ori_input_data
# input_data = MinMaxScaler().fit_transform(ori_input_data)
input_data = StandardScaler().fit_transform(ori_input_data)

cfg['n_clusters'] = len(np.unique(labels))
cfg['use_kmeans'] = True
mresults = []
mresults_idx = 0

time_base = time.time()

cfg['dim_reduction'] = 'pass'  # pass / dec
if cfg['dim_reduction']=='pass':
    output_data = input_data
elif cfg['dim_reduction']=='load':
    output_data = pd.read_excel('./data/'+cfg['dset']+'/dec_data.xlsx', sheet_name='data', header=None).to_numpy()
    labels = pd.read_excel('./data/'+cfg['dset']+'/dec_data.xlsx', sheet_name='label', header=None).to_numpy().flatten()
elif cfg['dim_reduction'] == 'dec':
    label_pred, output_data = dec_reduction(input_data, labels, cfg)
    with pd.ExcelWriter('./data/'+cfg['dset']+'/dec_data.xlsx') as writer:
        data_pd = pd.DataFrame(output_data)
        label_pd = pd.DataFrame(labels)
        data_pd.to_excel( writer, sheet_name='data', header=False, index=False)
        label_pd.to_excel( writer, sheet_name='label', header=False, index=False)

cfg['use_siamese'] = 'pass'  # load / pass / train
if cfg['use_siamese'] == 'train':
    input_data = []
    input_data = output_data
    output_data = []
    epochs = cfg['siam_ne']
    input_shape = input_data.shape[1:]
    for i in range(1, 2):
    # create training+test positive and negative pairs
        cfg['siam_k'] = i * 2
        cfg['siamese_tot_pairs'] = 200000
        tr_pairs, tr_y = create_pairs_from_unlabeled_data(input_data, None, cfg.get('siam_k'), cfg.get('siamese_tot_pairs'), False)
        te_pairs= tr_pairs
        te_y = tr_y
        siamesenet = SiameseNetwork(cfg, input_shape)
        history = siamesenet.train(tr_pairs, tr_y, te_pairs, te_y, epochs)
        output_data = siamesenet.predict(input_data, cfg['batch_size'])
        siamdataset = (output_data, labels), (output_data, labels)
        if i == 1:
            with pd.ExcelWriter('./data/'+cfg['dset']+'/siam_data.xlsx', mode='w', engine='openpyxl') as writer:
                sheet_nm = 'data' + str(i)
                data_pd = pd.DataFrame(output_data)
                data_pd.to_excel(writer, sheet_name=sheet_nm, header=False, index=False)
        else:
            with pd.ExcelWriter('./data/'+cfg['dset']+'/siam_data.xlsx', mode='a', engine='openpyxl') as writer:
                sheet_nm = 'data' + str(i)
                data_pd = pd.DataFrame(output_data)
                data_pd.to_excel(writer, sheet_name=sheet_nm, header=False, index=False)
        mresult, predictions = get_results(output_data, labels, cfg)
        mresults.append(mresult)
    with pd.ExcelWriter('./data/' + cfg['dset'] + '/siam_data.xlsx', mode='a', engine='openpyxl') as writer:
        label_pd = pd.DataFrame(labels)
        label_pd.to_excel(writer, sheet_name='label', header=False, index=False)
    mresults_idx = np.argmax(np.array([sum(mresults[i]) for i in range(0, len(mresults))]))
    mresult = mresults[mresults_idx]
elif cfg['use_siamese'] == 'load':
    cfg['siam_k'] = 2
    sheet_nm = 'data' + str(cfg['siam_k'])
    output_data = []
    output_data = pd.read_excel('./data/'+cfg['dset']+'/siam_data.xlsx', sheet_name=sheet_nm, header=None).to_numpy()
    labels = pd.read_excel('./data/' + cfg['dset'] + '/siam_data.xlsx', sheet_name='label', header=None).to_numpy().flatten()
    siamdataset = (output_data, labels), (output_data, labels)
elif cfg['use_siamese'] == 'pass':
    siamdataset = (output_data, labels), (output_data, labels)

output_data = []
cfg['data_dim'] = np.shape(siamdataset[0][0])[-1]
data_loader = DataLoader(cfg, siamdataset)
spectralnet = SpectralNet(cfg)
mmmodel = spectralnet.train(data_loader)
output_data = spectralnet.predict(data_loader)
mresult, predictions = get_results(output_data, labels, cfg)

totaltime = time.time() - time_base

predictions[labels_indexs] = predictions
ineedtosaveresults('cfg:', cfg, cfg['dset']+'.txt')
ineedtosaveresults('index:', mresults_idx, cfg['dset']+'.txt')
ineedtosaveresults('result:', mresult, cfg['dset']+'.txt')
ineedtosaveresults('time:', totaltime, cfg['dset']+'.txt')
print(mresult)
print(totaltime)