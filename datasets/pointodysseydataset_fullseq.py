from numpy import random
import torch
import numpy as np
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from torch._C import dtype, set_flush_denormal
import utils.basic
import utils.misc
import utils.improc
import glob
import cv2
from torchvision.transforms import ColorJitter, GaussianBlur
import albumentations as A
from functools import partial
import sys

class PointOdysseyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location='/orion/group/point_odyssey',
                 dset='test',
                 N=32,
                 verbose=False,
    ):
        print('loading pointodyssey fullseq dataset...')

        # in this version we load the full video,
        # rather than chopping it into subseqs of length S
        # and since this does not fit in memory,
        # we return paths instead of rgbs

        self.dset = dset
        self.N = N

        self.seq_paths = []
        self.rgb_paths = []
        self.traj_paths = []
        self.annotation_paths = []
        self.full_idxs = []

        self.subdirs = []
        self.sequences = []

        self.subdirs.append(os.path.join(dataset_location, dset))

        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*/")):
                seq_name = seq.split('/')[-1]
                self.sequences.append(seq)

        self.sequences = sorted(self.sequences)
        print('sequences', self.sequences)
        
        if verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))
        
        print('loading trajectories...')
        for seq in self.sequences:
            rgb_path = os.path.join(seq, 'rgbs')
            annotations_path = os.path.join(seq, 'anno.npz')
            assert(os.path.isfile(annotations_path))

            full_idx = np.arange(len(os.listdir(rgb_path)))
            self.rgb_paths.append([os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % (idx)) for idx in full_idx])
            self.annotation_paths.append(os.path.join(seq, 'anno.npz'))
            self.full_idxs.append(full_idx)

            if verbose:
                sys.stdout.write('.')
                sys.stdout.flush()
            else:
                if verbose:
                    sys.stdout.write('v')
                    sys.stdout.flush()

        print('collected %d videos in %s (dset=%s)' % (
            len(self.rgb_paths), dataset_location, dset))

    def __getitem__(self, index):
        seq = self.sequences[index]
        rgb_paths = self.rgb_paths[index]
        full_idx = self.full_idxs[index]

        print('seq', seq)
        
        annotations_path = self.annotation_paths[index]
        annotations = np.load(annotations_path, allow_pickle=True)
        trajs = annotations['trajs_2d'][full_idx].astype(np.float32)
        visibs = annotations['visibilities'][full_idx].astype(np.float32)
        valids = (visibs<2).astype(np.float32)
        visibs = (visibs==1).astype(np.float32)

        S,N,D = trajs.shape
        assert(D==2)

        valids_xy = np.ones_like(trajs)

        # some data is valid in 3d but invalid in 2d
        # here we will filter to the data which is valid in 2d
        valids_xy = np.ones_like(trajs)
        inf_idx = np.where(np.isinf(trajs))
        trajs[inf_idx] = 0
        valids_xy[inf_idx] = 0
        nan_idx = np.where(np.isnan(trajs))
        trajs[nan_idx] = 0
        valids_xy[nan_idx] = 0
        inv_idx = np.where(np.sum(valids_xy, axis=2)<2) # S,N
        visibs[inv_idx] = 0
        valids[inv_idx] = 0
        
        H, W = 540, 960
        
        # update visibility annotations
        for si in range(S):
            # avoid 1px edge
            oob_inds = np.logical_or(
                np.logical_or(trajs[si,:,0] < 1, trajs[si,:,0] > W-2),
                np.logical_or(trajs[si,:,1] < 1, trajs[si,:,1] > H-2))
            visibs[si,oob_inds] = 0
            # exclude oob from eval
            valids[si,oob_inds] = 0

        # ensure that the point is good at frame0
        vis_and_val = valids * visibs
        vis0 = vis_and_val[0] > 0
        trajs = trajs[:,vis0]
        visibs = visibs[:,vis0]
        valids = valids[:,vis0]
        print('trajs vis0', trajs.shape)
        
        # ensure that the point is good in at least K frames total
        K = 8
        vis_and_val = valids * visibs
        val_ok = np.sum(vis_and_val, axis=0) >= K
        trajs = trajs[:,val_ok]
        visibs = visibs[:,val_ok]
        valids = valids[:,val_ok]
        print('trajs vis%d' % K, trajs.shape)

        # ensure that the per-frame motion isn't too crazy
        mot = np.max(np.linalg.norm(trajs[1:] - trajs[:-1], axis=-1), axis=0) # N
        mot_ok = mot < 512
        trajs = trajs[:,mot_ok]
        visibs = visibs[:,mot_ok]
        valids = valids[:,mot_ok]
        print('trajs mot', trajs.shape)
        
        N = trajs.shape[1]

        assert(N > 0) # otw somehow all trajs got filtered out

        # we won't supervise with the extremes, but let's clamp anyway just to be safe
        trajs = np.minimum(np.maximum(trajs, np.array([-32,-32])), np.array([W+32, H+32])) # S,N,2
        
        N = trajs.shape[1]
        N_ = min(N, self.N)
        # inds = np.random.choice(N, N_, replace=False)
        inds = np.linspace(0, N-1, N_).astype(np.int32)

        trajs = trajs[:,inds]
        visibs = visibs[:,inds]
        valids = valids[:,inds]

        trajs = torch.from_numpy(trajs)  # S, N, 2
        visibs = torch.from_numpy(visibs)  # S, N
        valids = torch.from_numpy(valids)  # S, N

        sample = {
            'seq': seq,
            'rgb_paths': rgb_paths,
            'trajs': trajs,
            'visibs': visibs,
            'valids': valids,
        }
        return sample

    def __len__(self):
        return len(self.rgb_paths)
