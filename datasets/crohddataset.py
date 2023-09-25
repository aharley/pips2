import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
import glob
import json
import imageio
import cv2

class CrohdDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location,
                 S=1000,
                 S_min=128):
        self.S = S
        
        dataset_dir = '%s/HT21' % dataset_location
        label_location = '%s/HT21Labels' % dataset_location
        subfolders = []

        dataset_dir = os.path.join(dataset_dir, "train")
        label_location = os.path.join(label_location, "train")
        subfolders = ['HT21-01', 'HT21-02', 'HT21-03', 'HT21-04']
        
        print("loading data from {0}".format(dataset_dir))
        # read gt for subfolders
        self.dataset_dir = dataset_dir
        # self.seqlen = seqlen
        self.subfolders = subfolders
        self.folder_to_gt = {} # key: folder name, value: dict with fields boxlist, scorelist, vislist
        # self.subfolder_lens = []

        self.all_folders = []
        self.all_start_frames = []
        self.all_end_frames = []
        self.all_dicts = []

        # we'll take each length-1k subseq as a video
        
        for fid, subfolder in enumerate(subfolders):
            print("loading labels for folder {0}/{1}".format(fid+1, len(subfolders)))
            label_path = os.path.join(dataset_dir, subfolder, 'gt/gt.txt')
            labels = np.loadtxt(label_path, delimiter=',')

            n_frames = int(labels[-1,0])
            # print('n_frames', n_frames)
            # print('n_frames//seqlen', n_frames//seqlen)
            
            n_heads = int(labels[:,1].max())

            boxlist = np.zeros((n_frames, n_heads, 4))
            scorelist = -1 * np.ones((n_frames, n_heads))
            vislist = np.zeros((n_frames, n_heads))

            # print('labels', labels.shape)

            for i in range(labels.shape[0]):
                frame_id, head_id, bb_left, bb_top, bb_width, bb_height, conf, cid, vis = labels[i]
                frame_id = int(frame_id) - 1 # convert 1 indexed to 0 indexed
                head_id = int(head_id) - 1 # convert 1 indexed to 0 indexed

                scorelist[frame_id, head_id] = 1
                vislist[frame_id, head_id] = vis
                box_cur = np.array([bb_left, bb_top, bb_left+bb_width, bb_top+bb_height]) # convert xywh to x1, y1, x2, y2
                boxlist[frame_id, head_id] = box_cur

            d = {
                'boxlist': np.copy(boxlist),
                'scorelist': np.copy(scorelist),
                'vislist': np.copy(vislist)
            }

            for start_frame in range(0, n_frames, self.S//2):
                end_frame = min(start_frame+self.S, n_frames)
                
                if end_frame-start_frame >= S_min: 
                    print('%d:%d' % (start_frame, end_frame))
                    self.all_folders.append(subfolder)
                    self.all_start_frames.append(start_frame)
                    self.all_end_frames.append(end_frame)
                    self.all_dicts.append(d)

        print('found %d samples' % len(self.all_folders))

    def __getitem__(self, index):
        print('index', index)

        folder = self.all_folders[index]
        d = self.all_dicts[index]
        start_frame = self.all_start_frames[index]
        end_frame = self.all_end_frames[index]

        S = self.S
        boxlist = d['boxlist'][start_frame:end_frame]
        valids = d['scorelist'][start_frame:end_frame]
        visibs = d['vislist'][start_frame:end_frame]
        # print('boxlist', boxlist.shape)
        
        # # get gt
        # boxlist = self.folder_to_gt[subfolder]['boxlist'][start_frame:start_frame+S] / 2 # S, n_head, 4
        # scorelist = self.folder_to_gt[subfolder]['scorelist'][start_frame:start_frame+S] # S, n_head
        # vislist = self.folder_to_gt[subfolder]['vislist'][start_frame:start_frame+S] # S, n_head
        # # print("boxlist", boxlist.shape, "scorelist", scorelist.shape, "vislist", vislist.shape)
        # self.rgb_paths = []

        # rgb_paths = []
        # for ii in range(start_frame, end_frame):
        #     rgb_path = os.path.join(self.dataset_dir, folder, 'img1', str(start_frame+1).zfill(6)+'.jpg')
        #     rgb_paths.append(rgb_path)

        # rgbs = []
        # for i in range(len(boxlist)):
        #     # read image
        #     image_name = os.path.join(self.dataset_dir, subfolder, 'img1', str(start_frame+i+1).zfill(6)+'.jpg')
        #     rgb = Image.open(image_name)
        #     # downsample
        #     rgb = rgb.resize((int(rgb.size[0]/2), int(rgb.size[1]/2)), Image.BILINEAR)
        #     rgb = np.array(rgb)
        #     rgbs.append(rgb)
        # rgbs = np.stack(rgbs, axis=0)

        print('N orig', boxlist.shape[1])
        
        trajs = np.stack([boxlist[:, :, [0,2]].mean(2), boxlist[:, :, [1,3]].mean(2)], axis=2) # S,N,2
        print('trajs', trajs.shape)
        print('visibs', visibs.shape)
        print('valids', valids.shape)

        # assert(trajs.shape[0]==self.S)

        S, N = trajs.shape[:2]
        H, W = 1080, 1920
        # update visibility annotations
        for si in range(S):
            # crohd annotations get noisy/wrong near edges
            oob_inds = np.logical_or(
                np.logical_or(trajs[si,:,0] < 32, trajs[si,:,0] > W-32),
                np.logical_or(trajs[si,:,1] < 32, trajs[si,:,1] > H-32))
            visibs[si,oob_inds] = 0
            # exclude oob from eval
            valids[si,oob_inds] = 0
        
        # _, _, _, H, W = rgbs.shape
        # print('H', H, 'W', W)
        # for s in range(S):
        #     for n in range(kp_xys.shape[2]):
        #         xy = kp_xys[0, s, n]
        #         if xy[0] <= 0 or xy[0] >= W or xy[1] <= 0 or xy[1] >= H:
        #             valids[0, s, n] = 0

        # N = trajs.shape[1]
        # H, W = 1080, 1920
        # for s in range(S):
        #     for n in range(N):
        #         x, y = trajs[s,n,0], trajs[s,n,1]
        #         if x <= 0 or x >= W-1 or y <= 0 or y >= H-1:
        #             valids[s,n] = 0

        vis_ok0 = visibs[0] > 0 # N
        vis_ok1 = visibs[1] > 0 # N
        score_ok0 = valids[0] > 0 # N
        score_ok1 = valids[1] > 0 # N
        mot_ok = np.sum(np.linalg.norm(trajs[1:] - trajs[:-1], axis=-1), axis=0) > 150 # N
        all_ok = vis_ok0 * vis_ok1 * score_ok0 * score_ok1 * mot_ok
        print('np.sum(all_ok)', np.sum(all_ok))

        trajs = trajs[:,all_ok]
        visibs = visibs[:,all_ok]
        valids = valids[:,all_ok]

        S, N = trajs.shape[:2]
        # to improve the visuals, let's avoid shooting the traj oob
        for ni in range(N):
            for si in range(1,S):
                if visibs[si,ni]==0:
                    trajs[si,ni] = trajs[si-1,ni]

        sample = {
            # 'rgb_paths': rgb_paths,
            'folder': os.path.join(self.dataset_dir, folder),
            'start_frame': start_frame,
            'end_frame': end_frame,
            'trajs': trajs,
            'visibs': visibs,
            'valids': valids,
        }
        return sample

    def __len__(self):
        # print("subfolder_lens", self.subfolder_lens)
        # return sum(self.subfolder_lens)
        return len(self.all_folders)

if __name__ == "__main__":
    B = 1
    S = 1200
    shuffle=False
    dataset = CrohdDataset('/Users/yangzheng/code/project/long-term-tracking/data/HT21', seqlen=S, dset='t')

    from torch.utils.data import Dataset, DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=0)

    print(len(train_dataloader))
    train_iterloader = iter(train_dataloader)

    sample = next(train_iterloader)
    print(sample['rgbs'].shape)

