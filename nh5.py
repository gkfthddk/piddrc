import os
import sys
import torch as th
import numpy as np
import random
import h5py
import yaml
import tqdm
import time
import datetime

from collections import namedtuple
def re_namedtuple(pd):
  if isinstance(pd,dict):
    for key in pd:
      if isinstance(pd[key],dict):
        pd[key] = re_namedtuple(pd[key])
    pod = namedtuple('cfg',pd.keys())
    bobo = pod(**pd)
  return bobo
""" Example data:
  DRcalo3dHits.energy 14.007626 0.0
  DRcalo3dHits.ix 54 0
  DRcalo3dHits.iy 54 0
  DRcalo3dHits.noPhi 282 0
  DRcalo3dHits.noTheta 91 -92
  DRcalo3dHits.time 84.9 0.0
  DRcalo3dHits.type 1 0
  E_gen 100 1
  GenParticles.PDG 211 11
  GenParticles.momentum.p 109.96124267578125 10.000219345092773
  GenParticles.momentum.phi 0.22193138301372528 -0.22198565785390784
  GenParticles.momentum.theta 1.7149051427841187 0.20454877614974976
  Leakages.PDG 1000010030 -2212
  Leakages.momentum.p 60.201799783990815 0.0
  Leakages.momentum.phi 1.5707930326461792 -1.5707961320877075
  Leakages.momentum.theta 3.1409947872161865 0.0
  SimCalorimeterHits.cellID 2318304 0
  SimCalorimeterHits.energy 93.972984 0.0
  SimCalorimeterHits.position.x 3800.0266 -3799.7925
  SimCalorimeterHits.position.y 3799.968 -3799.968
  SimCalorimeterHits.position.z 4544.4907 -4544.4907
"""
def apply_jitter(batch_data,is_point=True): 
    """ Randomly jitter points. jittering is per point. 
    """ 
    sigma=batch_data.shape[-1]*np.mean(batch_data)/10
    clip=5*sigma
    #assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(*batch_data.shape), -1 * clip, clip)
    if(is_point):
        jittered_data[:,:,0]=np.round(jittered_data[:,:,0])
        jittered_data[:,:,1]=np.round(jittered_data[:,:,1])
        jittered_data[:,:,4]=np.round(jittered_data[:,:,4])
    jittered_data += batch_data
    return jittered_data

class CombinedDataset(th.utils.data.Dataset):
    def __init__(self, cand,path=None,balance=None,config=None,is_train=False,is_val=False,is_test=False):
        self.dataset0 = h5pyData(cand,is_train=is_train,is_val=is_val,path=path,balance=balance,config=config.input0)
        self.dataset1 = h5pyData(cand,is_train=is_train,is_val=is_val,path=path,balance=balance,config=config.input1)
        self.dataset2 = h5pyData(cand,is_train=is_train,is_val=is_val,path=path,balance=balance,config=config.input2)
        self.is_test=is_test

    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2), len(self.dataset0))  # or assert equal

    def __getitem__(self, idx):
        if(not self.is_test):
            x0, _01, _02 = self.dataset0[idx]
            x1, y1, y2 = self.dataset1[idx]
            x2, _1, _2 = self.dataset2[idx]
            return x0, x1, x2, y1, y2
        else:
            x0, _01, _02, _03 = self.dataset0[idx]
            x1, y1, y2, ori = self.dataset1[idx]
            x2, _1, _2, _3 = self.dataset2[idx]
            return x0, x1, x2, y1, y2, ori


class h5pyData(th.utils.data.Dataset):
    def __init__(self, cand, is_train=False, is_val=False, path=None, balance=None, config=None):
        if path is None:
            path = "/users/yulee/torch/h5s"

        self.cand = cand
        self.target = config.target
        self.target_scale = np.array(config.target_scale)
        self.num_point = config.num_point if config.num_point != "None" else None
        self.channel = config.channel
        self.xmax = config.channelmax
        self.num_ch = len(self.channel)
        self.is_train = is_train
        self.is_val = is_val

        # --- Augmentation setup ---
        self.flip = 'theta' in self.target
        if self.flip:
            self.THETA_SCALE = self.target_scale[self.target.index('theta')]
        self.noise = False

        self.h5_paths = [f'{path}/{name}.h5py' for name in self.cand]
        self.siglabel = []
        for i in range(len(self.cand)):
            label = [0.0] * len(self.cand)
            label[i] = 1.0
            self.siglabel.append(th.tensor(label).float())

        # --- Build Index Map ---
        self.index_map = []
        entries = []
        print("Building index map...")
        for i, h5_path in enumerate(self.h5_paths):
            try:
                with h5py.File(h5_path, 'r') as f:
                    # Use a smaller, faster-to-read dataset for entry count
                    num_entries = len(f['E_gen'])
                    entries.append(num_entries)
            except (IOError, KeyError) as e:
                print(f"Warning: Could not read {h5_path}. Skipping. Error: {e}")
                entries.append(0)

        self.balance = min(entries) if balance is None else min([balance] + entries)

        # This will be initialized in each worker process, so set to None here.
        self.file_handles = None

        self.index_map = []
        kf = config.kf if hasattr(config, 'kf') else 2

        for k in tqdm.tqdm(range(kf), desc=f"Mapping indices, balance:{self.balance}"):
            for j, h5_path in enumerate(self.h5_paths):
                bbin = self.balance / kf
                start = self.balance / kf * k # Corrected from i to k

                if is_train:
                    begin, end = int(start), int(start + bbin * 0.7 * 0.7)
                elif is_val:
                    begin, end = int(start + bbin * 0.7 * 0.7), int(start + bbin * 0.7)
                else: # is_test
                    begin, end = int(start + bbin * 0.7), int(start + bbin)

                with h5py.File(h5_path, 'r') as f:
                    mask = f['C_amp'][begin:end] + f['S_amp'][begin:end]
                    valid_indices = np.where(mask <= 2e+8)[0] + begin
                
                for h5_idx in valid_indices:
                    self.index_map.append((j, h5_idx))

    def __del__(self):
        if self.file_handles:
            for fh in self.file_handles:
                fh.close()

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        # Open file handles if they are not already open (worker-local)
        if self.file_handles is None:
            self.file_handles = [h5py.File(p, 'r') for p in self.h5_paths]

        file_idx, h5_idx = self.index_map[index]
        fh = self.file_handles[file_idx]

        # --- Load X data ---
        if self.num_point is not None:
            X_buf = np.zeros((self.num_point, self.num_ch), dtype=np.float32)
            for i, ch_name in enumerate(self.channel):
                dataset = fh[ch_name]
                if len(dataset.shape) == 2:
                    X_buf[:, i] = dataset[h5_idx, :self.num_point]
                elif len(dataset.shape) == 1 and self.num_point == 1:
                    X_buf[0, i] = dataset[h5_idx]
        else:
            X_buf = np.array([fh[ch][h5_idx] for ch in self.channel], dtype=np.float32).T

        # --- Load Y data ---
        Y1_buf = self.siglabel[file_idx]
        if len(self.target) > 0:
            Y2_buf = np.array([fh[t][h5_idx] for t in self.target], dtype=np.float32)
            Y2_buf *= self.target_scale
        else:
            Y2_buf = np.array([], dtype=np.float32)

        # --- Normalization (on-the-fly) ---
        for i in range(self.num_ch):
            if self.xmax[i] is not None and not np.isclose(self.xmax[i], 1.0) and not np.isclose(self.xmax[i], 0.0):
                if self.num_point is not None:
                    X_buf[:, i] /= self.xmax[i]
                else:
                    # This case seems to be for when num_point is None, where X_buf is (num_ch,).
                    X_buf[i] /= self.xmax[i]

        # --- Augmentation (optional) ---
        if self.is_train and self.flip and random.random() > 0.5:
            theta_channel_index = 1 # Assuming 'theta' is the second channel
            theta_target_index = self.target.index('theta')
            X_buf[:, theta_channel_index] *= -1
            Y2_buf[theta_target_index] = np.pi - Y2_buf[theta_target_index] + 0.006 * self.THETA_SCALE

        X_tensor = th.from_numpy(X_buf).float()
        Y2_tensor = th.from_numpy(Y2_buf).float()

        if self.is_train or self.is_val:
            return X_tensor, Y1_buf, Y2_tensor
        else: # is_test
            origin = th.tensor([file_idx, h5_idx]).float()
            return X_tensor, Y1_buf, Y2_tensor, origin

#normalize()
#standardscale()

class RandomFlip(object):
    def __init__(self,axis=[0,1]):
        self.axis=axis
    def __call__(self,x):
        # work on a clone to avoid modifying the original tensor in-place
        coords = x.clone()
        b = coords.size()[0]
        for i in range(b):
            if random.random() < 0.95:
                for ax in self.axis:
                    if random.random() < 0.5:
                        coords[i, :, ax] = -coords[i, :, ax]
        return coords
        


if __name__== '__main__':
    CHANNEL=[
                'DRcalo3dHits.amplitude_sum',
                'DRcalo3dHits.type',
                'DRcalo3dHits.time',
                'DRcalo3dHits.position.x',
                'DRcalo3dHits.position.y',
                'DRcalo3dHits.position.z',
    ]
    CHANNELMAX={
            'DRcalo3dHits.amplitude_sum':1,
            'DRcalo3dHits.type':1,
            'DRcalo3dHits.time':1,
            'DRcalo3dHits.position.x':1,
            'DRcalo3dHits.position.y':1,
            'DRcalo3dHits.position.z':1,
    }
    channel=CHANNEL[:3]
    channelmax=[CHANNELMAX[c] for c in channel]
    config=f"""
batch_size: 64
num_point: 2048
input_channel: 3
channel: {channel}
channelmax: {channelmax}
target: ["C_amp","S_amp"]
target_scale: [0.03,0.003]
cls_dim: 2
emb_dim: 32
depth: 2
num_head: 4
group_size: 16
num_group: 16
trans_dim: 32
encoder_dims: 32
encoder_feature: 32
fine_dim: 128
rms_norm: False
drop_path: 0.2
drop_out: 0.1
"""
    cand=["e-_20GeV","pi+_20GeV"]
    balance=1000
    cfg=re_namedtuple(yaml.safe_load(config))
    batch_size=cfg.batch_size
    train_dataset=h5pyData(cand,is_train=True,balance=balance,config=cfg)
    #train_loader=th.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=len(cand),drop_last=True,shuffle=True)
    train_loader=th.utils.data.DataLoader(train_dataset,batch_size=batch_size,drop_last=True,shuffle=True)
    print(len(train_loader))
    X,Y1,Y2=next(iter(train_loader))
    print("Sample shapes:", X.shape, Y1.shape, Y2.shape)
    print("Sample Y1:", Y1)
    print("Sample Y2:", Y2)
    cand=["e-_20GeV","pi+_20GeV"]
    balance=1000
    cfg=re_namedtuple(yaml.safe_load(config))
    batch_size=cfg.batch_size
    train_dataset=h5pyData(cand,is_train=True,balance=balance,config=cfg)
    #train_loader=th.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=len(cand),drop_last=True,shuffle=True)
    train_loader=th.utils.data.DataLoader(train_dataset,batch_size=batch_size,drop_last=True,shuffle=True)
    print(len(train_loader))
    X,Y1,Y2=next(iter(train_loader))
    print("Sample shapes:", X.shape, Y1.shape, Y2.shape)
    print("Sample Y1:", Y1)
    print("Sample Y2:", Y2)
