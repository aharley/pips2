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
                 strides=[1,2,4],
                 clip_step=2,
                 quick=False,
                 verbose=False,
    ):
        print('loading pointodyssey dataset...')

        self.S = S
        self.N = N
        self.verbose = verbose

        self.use_augs = use_augs
        self.dset = dset

        self.rgb_paths = []
        self.depth_paths = []
        self.normal_paths = []
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
        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))
        
        ## load trajectories
        print('loading trajectories...')

        if quick:
           self.sequences = self.sequences[:1] 
        
        for seq in self.sequences:
            if self.verbose: 
                print('seq', seq)

            rgb_path = os.path.join(seq, 'rgbs')
            info_path = os.path.join(seq, 'info.npz')
            annotations_path = os.path.join(seq, 'anno.npz')
            
            if os.path.isfile(info_path) and os.path.isfile(annotations_path):

                info = np.load(info_path, allow_pickle=True)
                trajs_3d_shape = info['trajs_3d'].astype(np.float32)

                if len(trajs_3d_shape) and trajs_3d_shape[1] > self.N:
                
                    for stride in strides:
                        for ii in range(0,len(os.listdir(rgb_path))-self.S*stride+1, clip_step):
                            full_idx = ii + np.arange(self.S)*stride
                            self.rgb_paths.append([os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % idx) for idx in full_idx])
                            self.depth_paths.append([os.path.join(seq, 'depths', 'depth_%05d.png' % idx) for idx in full_idx])
                            self.normal_paths.append([os.path.join(seq, 'normals', 'normal_%05d.jpg' % idx) for idx in full_idx])
                            self.annotation_paths.append(os.path.join(seq, 'anno.npz'))
                            self.full_idxs.append(full_idx)
                        if self.verbose:
                            sys.stdout.write('.')
                            sys.stdout.flush()
                elif self.verbose:
                    print('rejecting seq for missing 3d')
            elif self.verbose:
                print('rejecting seq for missing info or anno')

        print('collected %d clips of length %d in %s (dset=%s)' % (
            len(self.rgb_paths), self.S, dataset_location, dset))

    def getitem_helper(self, index):
        sample = None
        gotit = False

        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        normal_paths = self.normal_paths[index]
        full_idx = self.full_idxs[index]
        annotations_path = self.annotation_paths[index]
        annotations = np.load(annotations_path, allow_pickle=True)
        trajs_2d = annotations['trajs_2d'][full_idx].astype(np.float32)
        visibs = annotations['visibs'][full_idx].astype(np.float32)
        valids = annotations['valids'][full_idx].astype(np.float32)
        trajs_world = annotations['trajs_3d'][full_idx].astype(np.float32)
        pix_T_cams = annotations['intrinsics'][full_idx].astype(np.float32)
        cams_T_world = annotations['extrinsics'][full_idx].astype(np.float32)

        # ensure no weird/huge values 
        trajs_world_sum = np.sum(np.abs(trajs_world - trajs_world[0:1]), axis=(0,2))
        not_huge = trajs_world_sum < 100
        trajs_world = trajs_world[:,not_huge]
        trajs_2d = trajs_2d[:,not_huge]
        valids = valids[:,not_huge]
        visibs = visibs[:,not_huge]
        
        # ensure that the point is good in frame0
        vis_and_val = valids * visibs
        vis0 = vis_and_val[0] > 0
        trajs_2d = trajs_2d[:,vis0]
        trajs_world = trajs_world[:,vis0]
        visibs = visibs[:,vis0]
        valids = valids[:,vis0]

        S,N,D = trajs_2d.shape
        assert(D==2)
        assert(S==self.S)
        
        if N < self.N//2:
            print('returning before cropping: N=%d; need at least N=%d' % (N, self.N//2))
            return None, False
        
        trajs_cam = utils.geom.apply_4x4_py(cams_T_world, trajs_world)
        trajs_pix = utils.geom.apply_pix_T_cam_py(pix_T_cams, trajs_cam)

        # get rid of infs and nans in 2d
        valids_xy = np.ones_like(trajs_2d)
        inf_idx = np.where(np.isinf(trajs_2d))
        trajs_world[inf_idx] = 0
        trajs_cam[inf_idx] = 0
        trajs_2d[inf_idx] = 0
        valids_xy[inf_idx] = 0
        nan_idx = np.where(np.isnan(trajs_2d))
        trajs_world[nan_idx] = 0
        trajs_cam[nan_idx] = 0
        trajs_2d[nan_idx] = 0
        valids_xy[nan_idx] = 0
        inv_idx = np.where(np.sum(valids_xy, axis=2)<2) # S,N
        visibs[inv_idx] = 0
        valids[inv_idx] = 0

        rgbs = []
        for rgb_path in rgb_paths:
            with Image.open(rgb_path) as im:
                rgbs.append(np.array(im)[:, :, :3])

        depths = []
        for depth_path in depth_paths:
            depth16 = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            depth = depth16.astype(np.float32) / 65535.0 * 1000.0
            depths.append(depth)

        normals = []
        for normal_path in normal_paths:
            with Image.open(normal_path) as im:
                normals.append(np.array(im)[:, :, :3])

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
                np.logical_or(trajs_2d[si,:,0] < -64, trajs_2d[si,:,0] > W+64),
                np.logical_or(trajs_2d[si,:,1] < -64, trajs_2d[si,:,1] > H+64))
            valids[si,very_oob_inds] = 0

        # ensure that the point is good in frame0
        vis_and_val = valids * visibs
        vis0 = vis_and_val[0] > 0
        trajs_2d = trajs_2d[:,vis0]
        trajs_cam = trajs_cam[:,vis0]
        trajs_world = trajs_world[:,vis0]
        trajs_pix = trajs_pix[:,vis0]
        visibs = visibs[:,vis0]
        valids = valids[:,vis0]

        # ensure that the point is good in frame1
        vis_and_val = valids * visibs
        vis1 = vis_and_val[1] > 0
        trajs_2d = trajs_2d[:,vis1]
        trajs_cam = trajs_cam[:,vis1]
        trajs_world = trajs_world[:,vis1]
        trajs_pix = trajs_pix[:,vis1]
        visibs = visibs[:,vis1]
        valids = valids[:,vis1]

        # ensure that the point is good in at least sqrt(S) frames
        val_ok = np.sum(valids, axis=0) >= max(np.sqrt(S),2)
        trajs_2d = trajs_2d[:,val_ok]
        trajs_cam = trajs_cam[:,val_ok]
        trajs_world = trajs_world[:,val_ok]
        trajs_pix = trajs_pix[:,val_ok]
        visibs = visibs[:,val_ok]
        valids = valids[:,val_ok]
        
        N = trajs_2d.shape[1]
        
        if N < self.N//2:
            # print('N=%d' % (N))
            return None, False
        
        if N < self.N:
            print('N=%d; ideally we want N=%d, but we will pad' % (N, self.N))

        # even out the distribution, across initial positions and velocities
        # fps based on xy0 and mean motion
        xym = np.concatenate([trajs_2d[0], np.mean(trajs_2d[1:] - trajs_2d[:-1], axis=0)], axis=-1)
        inds = utils.misc.farthest_point_sample_py(xym, self.N)
        trajs_2d = trajs_2d[:,inds]
        trajs_cam = trajs_cam[:,inds]
        trajs_world = trajs_world[:,inds]
        trajs_pix = trajs_pix[:,inds]
        visibs = visibs[:,inds]
        valids = valids[:,inds]

        # we won't supervise with the extremes, but let's clamp anyway just to be safe
        trajs_2d = np.minimum(np.maximum(trajs_2d, np.array([-64,-64])), np.array([W+64, H+64])) # S,N,2
        trajs_pix = np.minimum(np.maximum(trajs_pix, np.array([-64,-64])), np.array([W+64, H+64])) # S,N,2
            
        N = trajs_2d.shape[1]
        N_ = min(N, self.N)
        inds = np.random.choice(N, N_, replace=False)

        # prep for batching, by fixing N
        trajs_2d_full = np.zeros((self.S, self.N, 2)).astype(np.float32)
        trajs_cam_full = np.zeros((self.S, self.N, 3)).astype(np.float32)
        trajs_world_full = np.zeros((self.S, self.N, 3)).astype(np.float32)
        trajs_pix_full = np.zeros((self.S, self.N, 2)).astype(np.float32)
        visibs_full = np.zeros((self.S, self.N)).astype(np.float32)
        valids_full = np.zeros((self.S, self.N)).astype(np.float32)
        trajs_2d_full[:,:N_] = trajs_2d[:,inds]
        trajs_cam_full[:,:N_] = trajs_cam[:,inds]
        trajs_world_full[:,:N_] = trajs_world[:,inds]
        trajs_pix_full[:,:N_] = trajs_pix[:,inds]
        visibs_full[:,:N_] = visibs[:,inds]
        valids_full[:,:N_] = valids[:,inds]

        rgbs = torch.from_numpy(np.stack(rgbs, 0)).permute(0,3,1,2) # S,3,H,W
        depths = torch.from_numpy(np.stack(depths, 0)).unsqueeze(1) # S,1,H,W
        normals = torch.from_numpy(np.stack(normals, 0)).permute(0,3,1,2) # S,3,H,W
        trajs_2d = torch.from_numpy(trajs_2d_full) # S,N,2
        trajs_cam = torch.from_numpy(trajs_cam_full) # S,N,3
        trajs_world = torch.from_numpy(trajs_world_full) # S,N,3
        trajs_pix = torch.from_numpy(trajs_pix_full) # S,N,2
        visibs = torch.from_numpy(visibs_full) # S,N
        valids = torch.from_numpy(valids_full) # S,N

        sample = {
            'rgbs': rgbs,
            'depths': depths,
            'normals': normals,
            'trajs_2d': trajs_2d,
            'trajs_cam': trajs_cam,
            'trajs_world': trajs_world,
            'trajs_pix': trajs_pix,
            'pix_T_cams': pix_T_cams,
            'cams_T_world': cams_T_world,
            'visibs': visibs,
            'valids': valids,
            'annotations_path': annotations_path,
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
                'depths': torch.zeros((self.S, 1, 540, 960)),
                'trajs_2d': torch.zeros((self.S, self.N, 2)),
                'trajs_cam': torch.zeros((self.S, self.N, 3)),
                'trajs_pix': torch.zeros((self.S, self.N, 2)),
                'visibs': torch.zeros((self.S, self.N)),
                'valids': torch.zeros((self.S, self.N)),
            }
        return sample, gotit

    def __len__(self):
        return len(self.rgb_paths)
