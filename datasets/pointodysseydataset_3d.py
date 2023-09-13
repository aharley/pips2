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
import utils.improc
import glob
import cv2
from torchvision.transforms import ColorJitter, GaussianBlur
import albumentations as A
from functools import partial
import sys

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

class PointOdysseyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location='/orion/group/point_odyssey',
                 dset='train',
                 use_augs=False,
                 S=8,
                 N=32,
                 quick=False,
                 verbose=False,
    ):
        print('loading pointodyssey dataset...')

        self.S = S
        self.N = N

        self.use_augs = use_augs
        self.dset = dset

        self.rgb_paths = []
        self.depth_paths = []
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
        if verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))
        
        ## load trajectories
        print('loading trajectories...')

        if quick:
           self.sequences = self.sequences[:1] 
        
        for seq in self.sequences:

            # if 'character1' not in seq:
            
            rgb_path = os.path.join(seq, 'rgbs')

            annotations_path = os.path.join(seq, 'annotations.npz')
            if os.path.isfile(annotations_path):

                if verbose: 
                    print('seq', seq)
                    
                for stride in [1]:
                    for ii in range(0,len(os.listdir(rgb_path))-self.S*stride+1, 4):
                        full_idx = ii + np.arange(self.S)*stride
                        self.rgb_paths.append([os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % idx) for idx in full_idx])
                        self.depth_paths.append([os.path.join(seq, 'depths', 'depth_%05d.png' % idx) for idx in full_idx])
                        self.annotation_paths.append(os.path.join(seq, 'annotations.npz'))
                        self.full_idxs.append(full_idx)
                        if verbose:
                            sys.stdout.write('.')
                            sys.stdout.flush()
                        else:
                            if verbose:
                                sys.stdout.write('v')
                                sys.stdout.flush()

        print('collected %d clips of length %d in %s (dset=%s)' % (
            len(self.rgb_paths), self.S, dataset_location, dset))

    def getitem_helper(self, index):
        sample = None
        gotit = False


        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        full_idx = self.full_idxs[index]
        annotations_path = self.annotation_paths[index]
        annotations = np.load(annotations_path, allow_pickle=True)
        # print(annotations.files)
        # print('annotations_path', annotations_path)
        # print('full_idx', full_idx)
        # print('raw trajs_2d', annotations['trajs_2d'].shape)
        # print('raw trajs_3d', annotations['trajs_3d'].shape)
        
        trajs_2d = annotations['trajs_2d'][full_idx].astype(np.float32)
        visibs = annotations['visibilities'][full_idx].astype(np.float32)
        valids = (visibs<2).astype(np.float32)
        visibs = (visibs==1).astype(np.float32)

        trajs_world = annotations['trajs_3d'][full_idx].astype(np.float32)
        pix_T_cams = annotations['intrinsics'][full_idx].astype(np.float32)
        cams_T_world = annotations['extrinsics'][full_idx].astype(np.float32)

        S,N,D = trajs_2d.shape
        assert(D==2)
        assert(S==self.S)
        
        trajs_cam = utils.geom.apply_4x4_py(cams_T_world, trajs_world)
        trajs_pix = utils.geom.apply_pix_T_cam_py(pix_T_cams, trajs_cam)

        valids_xy = np.ones_like(trajs_2d)

        # get rid of infs and nans
        inf_idx = np.where(np.isinf(trajs_2d))
        trajs_2d[inf_idx] = 0
        valids_xy[inf_idx] = 0
        nan_idx = np.where(np.isnan(trajs_2d))
        trajs_2d[nan_idx] = 0
        valids_xy[nan_idx] = 0

        inv_idx = np.where(np.sum(valids_xy, axis=2)<2) # S,N
        visibs[inv_idx] = 0
        valids[inv_idx] = 0

        rgbs = []
        for rgb_path in rgb_paths:
            # rgbs.append(cv2.imread(rgb_path))
            with Image.open(rgb_path) as im:
                rgbs.append(np.array(im)[:, :, :3])

        depths = []
        for depth_path in depth_paths:
            depth16 = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            depth = depth16.astype(np.float32) / 65535.0 * 1000.0
            depths.append(depth)

        H,W,C = rgbs[0].shape
        assert(C==3)

        H,W,C = rgbs[0].shape
        assert(C==3)
        
        # update visibility annotations
        for si in range(S):
            # avoid 1px edge
            oob_inds = np.logical_or(
                np.logical_or(trajs_2d[si,:,0] < 1, trajs_2d[si,:,0] > W-2),
                np.logical_or(trajs_2d[si,:,1] < 1, trajs_2d[si,:,1] > H-2))
            visibs[si,oob_inds] = 0

            # when a point moves far oob, don't supervise with it
            very_oob_inds = np.logical_or(
                np.logical_or(trajs_2d[si,:,0] < -32, trajs_2d[si,:,0] > W+32),
                np.logical_or(trajs_2d[si,:,1] < -32, trajs_2d[si,:,1] > H+32))
            valids[si,very_oob_inds] = 0

        # ensure that the point is valid at frame0
        val0 = valids[0] > 0
        trajs_2d = trajs_2d[:,val0]
        trajs_cam = trajs_cam[:,val0]
        trajs_pix = trajs_pix[:,val0]
        visibs = visibs[:,val0]
        valids = valids[:,val0]

        # ensure that the point is visible at frame0
        vis0 = visibs[0] > 0
        trajs_2d = trajs_2d[:,vis0]
        trajs_cam = trajs_cam[:,vis0]
        trajs_pix = trajs_pix[:,vis0]
        visibs = visibs[:,vis0]
        valids = valids[:,vis0]

        # ensure that the point is visible in at least sqrt(S) frames
        vis_ok = np.sum(visibs, axis=0) >= max(np.sqrt(S),2)
        trajs_2d = trajs_2d[:,vis_ok]
        trajs_cam = trajs_cam[:,vis_ok]
        trajs_pix = trajs_pix[:,vis_ok]
        visibs = visibs[:,vis_ok]
        valids = valids[:,vis_ok]
        
        N = trajs_2d.shape[1]
        
        if N==0:
            print('N=%d' % (N))
            return None, False
        
        if N < self.N:
            print('N=%d; ideally we want N=%d, but we will pad' % (N, self.N))

        if N > self.N*4:
            # fps based on position and motion
            xym = np.concatenate([np.mean(trajs_2d, axis=0), np.mean(trajs_2d[1:] - trajs_2d[:-1], axis=0)], axis=-1)
            inds = utils.misc.farthest_point_sample_py(xym, self.N*4)
            trajs_2d = trajs_2d[:,inds]
            trajs_cam = trajs_cam[:,inds]
            trajs_pix = trajs_pix[:,inds]
            visibs = visibs[:,inds]
            valids = valids[:,inds]
            
        N = trajs_2d.shape[1]
        N_ = min(N, self.N)
        inds = np.random.choice(N, N_, replace=False)

        # clamp so that the trajectories don't get too crazy
        trajs_2d = np.minimum(np.maximum(trajs_2d, np.array([-32,-32])), np.array([W+31, H+31])) # S,N,2
        trajs_pix = np.minimum(np.maximum(trajs_pix, np.array([-32,-32])), np.array([W+31, H+31])) # S,N,2
        
        trajs_2d_full = np.zeros((self.S, self.N, 2)).astype(np.float32)
        trajs_cam_full = np.zeros((self.S, self.N, 3)).astype(np.float32)
        trajs_pix_full = np.zeros((self.S, self.N, 2)).astype(np.float32)
        visibs_full = np.zeros((self.S, self.N)).astype(np.float32)
        valids_full = np.zeros((self.S, self.N)).astype(np.float32)

        trajs_2d_full[:,:N_] = trajs_2d[:,inds]
        trajs_cam_full[:,:N_] = trajs_cam[:,inds]
        trajs_pix_full[:,:N_] = trajs_pix[:,inds]
        visibs_full[:,:N_] = visibs[:,inds]
        valids_full[:,:N_] = valids[:,inds]

        rgbs = torch.from_numpy(np.stack(rgbs, 0)).permute(0,3,1,2) # S,3,H,W
        depths = torch.from_numpy(np.stack(depths, 0)).unsqueeze(1) # S,1,H,W
        trajs_2d = torch.from_numpy(trajs_2d_full) # S,N,2
        trajs_cam = torch.from_numpy(trajs_cam_full) # S,N,3
        trajs_pix = torch.from_numpy(trajs_pix_full) # S,N,2
        visibs = torch.from_numpy(visibs_full) # S,N
        valids = torch.from_numpy(valids_full) # S,N

        sample = {
            'rgbs': rgbs,
            'depths': depths,
            'trajs_2d': trajs_2d,
            'trajs_cam': trajs_cam,
            'trajs_pix': trajs_pix,
            'visibs': visibs,
            'valids': valids,
        }
        
        return sample, True

    
    def __getitem__(self, index):
        gotit = False
        sample, gotit = self.getitem_helper(index)
        if not gotit:
            print('warning: sampling failed')
            # return a fake sample, so we can still collate
            sample = {
                'rgbs': torch.zeros((self.S, 3, 540, 960)),
                'trajs': torch.zeros((self.S, self.N, 2)),
                'visibs': torch.zeros((self.S, self.N)),
                'valids': torch.zeros((self.S, self.N)),
            }
        return sample, gotit

    def __len__(self):
        return len(self.rgb_paths)
