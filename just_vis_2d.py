import time
import numpy as np
import timeit
import saverloader
import utils.improc
import utils.geom
import utils.misc
import random
from utils.basic import print_, print_stats
from datasets.pointodysseydataset import PointOdysseyDataset
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

random.seed(125)
np.random.seed(125)
torch.manual_seed(125)
    
def run_model(d, device, sw=None):
    rgbs = d['rgbs'].to(device).float() # B,S,C,H,W
    trajs_g = d['trajs'].to(device).float() # B,S,N,2
    vis_g = d['visibs'].to(device).float() # B,S,N
    valids = d['valids'].to(device).float() # B,S,N

    print_stats('rgbs', rgbs)
    print_stats('trajs_g', trajs_g)
    print_stats('vis_g', vis_g)
    print_stats('valids', valids)

    B, S, C, H, W = rgbs.shape
    assert(C==3)
    B, S, N, D = trajs_g.shape
    assert(D==2)
    
    if sw is not None and sw.save_this:

        prep_rgbs = utils.improc.preprocess_color(rgbs)
        prep_grays = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)

        sw.summ_traj2ds_on_rgb('0_inputs/trajs_g_on_rgb', trajs_g[0:1], prep_rgbs.mean(dim=1), valids=valids[0:1], cmap='winter')
        # sw.summ_traj2ds_on_rgbs('0_inputs/trajs_g_on_rgbs', trajs_g[0:1], prep_rgbs, valids=valids[0:1])
        sw.summ_traj2ds_on_rgbs2('0_inputs/trajs_g_on_rgbs2', trajs_g[0:1], vis_g[0:1], utils.improc.preprocess_color(rgbs[0:1]), valids=valids[0:1])

        # for the kp vis, we will clamp so that we can see everything
        trajs_g_clamp = trajs_g.clone()
        trajs_g_clamp[:,:,:,0] = trajs_g_clamp[:,:,:,0].clip(0,W-1)
        trajs_g_clamp[:,:,:,1] = trajs_g_clamp[:,:,:,1].clip(0,H-1)
        
        outs = sw.summ_pts_on_rgbs(
            '',
            trajs_g_clamp[0:1],
            prep_grays[0:1],
            valids=valids[0:1],
            cmap='winter', linewidth=3, only_return=True)
        sw.summ_pts_on_rgbs(
            '0_inputs/kps_gv_on_rgbs',
            trajs_g_clamp[0:1],
            utils.improc.preprocess_color(outs),
            valids=valids[0:1]*vis_g[0:1],
            cmap='spring', linewidth=2)

    return None 
    

def main(
        exp_name='debug',
        dset='train',
        B=1, # batchsize 
        S=32, # seqlen
        N=128, # number of points per clip
        crop_size=(256,448), 
        use_augs=False, # resizing/jittering/color/blur augs
        shuffle=False, # dataset shuffling
        log_dir='./logs_just_vis_2d',
        dataset_location='/orion/group/point_odyssey', 
        log_freq=1,
        max_iters=10,
        quick=False,
        dname=None,
):
    device = 'cpu:0'

    # the idea in this file is:
    # load pointodyssey data and visualize it
    
    exp_name = 'jv00' # copy from dev repo
    exp_name = 'jv01' # fix color bug...
    
    import socket
    host = socket.gethostname()

    assert(crop_size[0] % 32 == 0)
    assert(crop_size[1] % 32 == 0)
    
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
        use_augs=use_augs,
        crop_size=crop_size,
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
