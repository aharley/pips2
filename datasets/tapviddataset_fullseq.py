from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import pickle

class TapVidDavis(torch.utils.data.Dataset):
    def __init__(self, dataset_location='../datasets/tapvid_davis'):

        print('loading TAPVID-DAVIS dataset...')

        input_path = '%s/tapvid_davis.pkl' % dataset_location
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                data = list(data.values())
        self.data = data
        print('found %d videos in %s' % (len(self.data), dataset_location))
        
    def __getitem__(self, index):
        dat = self.data[index]
        rgbs = dat['video'] # list of H,W,C uint8 images
        trajs = dat['points'] # N,S,2 array
        valids = 1-dat['occluded'] # N,S array
        # note the annotations are only valid when not occluded
        
        trajs = trajs.transpose(1,0,2) # S,N,2
        valids = valids.transpose(1,0) # S,N

        vis_ok = valids[0] > 0
        trajs = trajs[:,vis_ok]
        valids = valids[:,vis_ok]

        # 1.0,1.0 should lie at the bottom-right corner pixel
        H, W, C = rgbs[0].shape
        trajs[:,:,0] *= W-1
        trajs[:,:,1] *= H-1

        rgbs = torch.from_numpy(np.stack(rgbs,0)).permute(0,3,1,2) # S,C,H,W
        trajs = torch.from_numpy(trajs) # S,N,2
        valids = torch.from_numpy(valids) # S,N

        sample = {
            'rgbs': rgbs,
            'trajs': trajs,
            'valids': valids,
            'visibs': valids,
        }
        return sample

    def __len__(self):
        return len(self.data)


