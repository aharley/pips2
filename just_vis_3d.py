import time
import numpy as np
import timeit
import saverloader
import utils.improc
import utils.geom
import utils.misc
import random
from utils.basic import print_, print_stats
from datasets.pointodysseydataset_3d import PointOdysseyDataset
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fire import Fire
import sys
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from torch.utils.data import Dataset, DataLoader
    
def run_model(d, device, sw=None):
    rgbs = d['rgbs'].to(device).float() # B,S,C,H,W
    depths = d['depths'].to(device).float() # B,1,C,H,W
    trajs_x = d['trajs_pix'].to(device).float() # B,S,N,2
    trajs_g = d['trajs_2d'].to(device).float() # B,S,N,2
    vis_g = d['visibs'].to(device).float() # B,S,N
    valids = d['valids'].to(device).float() # B,S,N

    print('rgbs', rgbs.shape)
    print('trajs_x', trajs_x.shape)
    print('trajs_g', trajs_g.shape)
    print('vis_g', vis_g.shape)
    print('valids', valids.shape, torch.sum(valids[:,0]))

    B, S, C, H, W = rgbs.shape
    assert(C==3)
    B, S, N, D = trajs_g.shape
    assert(D==2)

    print_stats('depths', depths)
    max_depth = 16
    depths_valid = (depths < max_depth).float() * (depths > 0.01).float()
    depths = depths * depths_valid
    print_stats('depths', depths)
    
    if sw is not None and sw.save_this:

        prep_rgbs = utils.improc.preprocess_color(rgbs)
        prep_grays = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)

        sw.summ_traj2ds_on_rgb('0_inputs/trajs_x_on_rgb', trajs_x[0:1], prep_rgbs.mean(dim=1), valids=valids[0:1], cmap='winter')
        sw.summ_traj2ds_on_rgb('0_inputs/trajs_g_on_rgb', trajs_g[0:1], prep_rgbs.mean(dim=1), valids=valids[0:1], cmap='winter')
        
    return None 
    

def main(
        exp_name='debug',
        dset='train',
        B=1, # batchsize 
        S=128, # seqlen
        N=256, # number of points per clip
        use_augs=False, # resizing/jittering/color/blur augs
        shuffle=False, # dataset shuffling
        log_dir='./logs_just_vis_3d',
        dataset_location='/orion/group/point_odyssey',
        log_freq=1,
        max_iters=10,
        # cuda
        device_ids=[0],
        quick=False,
        dname=None,
):
    device = 'cuda:%d' % device_ids[0]

    # the idea in this file is:
    # load the 3d pointodyssey data and visualize it
    
    exp_name = 'jw00' # copy from dev repo
    exp_name = 'jw01' # clean up

    # autogen a descriptive name
    model_name = "%d_%d_%d" % (B, S, N)
    if use_augs:
        model_name += "_A"
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    # get dataset
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    dataset_t = PointOdysseyDataset(
        dataset_location=dataset_location,
        dset=dset,
        S=S,
        N=N,
        quick=quick,
        verbose=True,
    )
    dataloader_t = DataLoader(
        dataset_t,
        batch_size=B,
        shuffle=shuffle,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        drop_last=True)
    iterloader_t = iter(dataloader_t)

    global_step = 0
    
    while global_step < max_iters:

        global_step += 1

        iter_start_time = time.time()
        iter_rtime = 0.0
        
        # read sample
        read_start_time = time.time()

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=min(S,8),
            scalar_freq=log_freq//5,
            just_gif=True)

        gotit = (False,False)
        while not all(gotit):
            try:
                sample, gotit = next(iterloader_t)
            except StopIteration:
                iterloader_t = iter(dataloader_t)
                sample, gotit = next(iterloader_t)

        rtime = time.time()-read_start_time
        iter_rtime += rtime

        _ = run_model(sample, device, sw=sw_t)

        iter_time = time.time()-iter_start_time
        
        print('%s; step %06d/%d; rtime %.2f; itime %.2f' % (
            model_name, global_step, max_iters, rtime, iter_time))
            
    writer_t.close()
            

if __name__ == '__main__':
    Fire(main)
