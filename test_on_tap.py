import time
import numpy as np
import timeit
import saverloader
from nets.pips2 import Pips
import utils.improc
import utils.geom
import utils.misc
import random
from utils.basic import print_, print_stats
from datasets.tapviddataset_fullseq import TapVidDavis
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

def create_pools(n_pool=1000):
    pools = {}

    thrs = [1,2,4,8,16]
    for thr in thrs:
        pools['d_%d' % thr] = utils.misc.SimplePool(n_pool, version='np')
        # pools['d_%d' % thr].update([0]) # conservative init
    pools['d_avg'] = utils.misc.SimplePool(n_pool, version='np')
    # pools['d_avg'].update([0]) # conservative init

    pool_names = [
        'l1',
        # 'l1_early',
        # 'ce',
        'l1_vis',
        'ate_all',
        'ate_vis',
        'ate_occ',
        'total_loss',
        'survival',
    ]
    for pool_name in pool_names:
        pools[pool_name] = utils.misc.SimplePool(n_pool, version='np')
    
    return pools


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def test_on_fullseq(model, d, sw, iters=8, S_max=8, image_size=(384,512)):
    metrics = {}

    rgbs = d['rgbs'].cuda().float() # B,S,C,H,W
    trajs_g = d['trajs'].cuda().float() # B,S,N,2
    valids = d['valids'].cuda().float() # B,S,N

    B, S, C, H, W = rgbs.shape
    B, S, N, D = trajs_g.shape
    assert(D==2)
    assert(B==1)

    rgbs_ = rgbs.reshape(B*S, C, H, W)
    H_, W_ = image_size
    sy = H_/H
    sx = W_/W
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    rgbs = rgbs_.reshape(B, S, 3, H_, W_)
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    H, W = H_, W_
    
    # zero-vel init
    trajs_e = trajs_g[:,0].repeat(1,S,1,1)
    
    cur_frame = 0
    done = False
    feat_init = None
    while not done:
        end_frame = cur_frame + S_max

        if end_frame > S:
            diff = end_frame-S
            end_frame = end_frame-diff
            cur_frame = max(cur_frame-diff,0)
        # print('working on subseq %d:%d' % (cur_frame, end_frame))

        traj_seq = trajs_e[:, cur_frame:end_frame]
        rgb_seq = rgbs[:, cur_frame:end_frame]
        S_local = rgb_seq.shape[1]

        if feat_init is not None:
            feat_init = [fi[:,:S_local] for fi in feat_init]
            
        preds, preds_anim, feat_init, _ = model(traj_seq, rgb_seq, iters=iters, feat_init=feat_init)

        trajs_e[:, cur_frame:end_frame] = preds[-1][:, :S_local]
        trajs_e[:, end_frame:] = trajs_e[:, end_frame-1:end_frame] # update the future

        if sw is not None and sw.save_this:
            traj_seq_e = preds[-1]
            traj_seq_g = trajs_g[:,cur_frame:end_frame]
            valid_seq = valids[:,cur_frame:end_frame]
            
            prep_rgbs = utils.improc.preprocess_color(rgb_seq)
            gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
            gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('', traj_seq_g, gray_rgbs[0:1].mean(dim=1), valids=valid_seq, cmap='winter', only_return=True))
            rgb_vis = []
            for tre in preds_anim:
                ate = torch.norm(tre - traj_seq_g, dim=-1) # B,S,N
                ate_all = utils.basic.reduce_masked_mean(ate, valid_seq, dim=[1,2]) # B
                rgb_vis.append(sw.summ_traj2ds_on_rgb('', tre[0:1], gt_rgb, valids=valid_seq, only_return=True, cmap='spring', frame_id=ate_all[0]))
            sw.summ_rgbs('3_test/animated_trajs_on_rgb_cur%02d' % cur_frame, rgb_vis)
        
        if end_frame >= S:
            done = True
        else:
            cur_frame = cur_frame + S_max - 1

    d_sum = 0.0
    thrs = [1,2,4,8,16]
    sx_ = W / 256.0
    sy_ = H / 256.0
    sc_py = np.array([sx_, sy_]).reshape([1,1,2])
    sc_pt = torch.from_numpy(sc_py).float().cuda()
    for thr in thrs:
        # note we exclude timestep0 from this eval
        d_ = (torch.norm(trajs_e[:,1:]/sc_pt - trajs_g[:,1:]/sc_pt, dim=-1) < thr).float() # B,S-1,N
        d_ = utils.basic.reduce_masked_mean(d_, valids[:,1:]).item()
        d_sum += d_
        metrics['d_%d' % thr] = d_
    d_avg = d_sum / len(thrs)
    metrics['d_avg'] = d_avg


    # print('trajs_g', trajs_g.shape)
    # print('trajs_e', trajs_e.shape)
    
    if sw is not None and sw.save_this:
        prep_rgbs = utils.improc.preprocess_color(rgbs)
        # label_colors = utils.improc.get_n_colors(N)
        # gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
        # outs = sw.summ_pts_on_rgbs(
        #     '', trajs_g[0:1],
        #     gray_rgbs[0:1],
        #     valids=valids[0:1],
        #     cmap='winter', linewidth=3, only_return=True)
        # sw.summ_pts_on_rgbs(
        #     'video_%d/kps_eg_on_rgbs' % (sw.global_step),
        #     trajs_e[0:1],
        #     utils.improc.preprocess_color(outs),
        #     cmap='spring', linewidth=3)

        rgb0 = sw.summ_traj2ds_on_rgb('', trajs_g[0:1], prep_rgbs[0:1,0], valids=valids[0:1], cmap='winter', linewidth=2, only_return=True)
        # sw.summ_traj2ds_on_rgb('0_inputs/%s_trajs_e0_on_rgb0' % dname, trajs_e0[0:1], utils.improc.preprocess_color(rgb0), valids=valids[0:1], cmap='spring', linewidth=2, frame_id=ate0_all[0].mean().item())
        sw.summ_traj2ds_on_rgb('2_outputs/trajs_e_on_rgb0', trajs_e[0:1], utils.improc.preprocess_color(rgb0), valids=valids[0:1], cmap='spring', linewidth=2, frame_id=d_avg*100.0)


        # for ni in range(N):
        #     rgb0_ = sw.summ_traj2ds_on_rgb('', trajs_g[0:1,:,ni:ni+1], prep_rgbs[0:1,0], valids=valids[0:1,:,ni:ni+1], cmap='winter', linewidth=2, only_return=True)
        #     sw.summ_traj2ds_on_rgb('2_outputs/trajs_e_on_rgb0_kp%02d' % ni, trajs_e[0:1,:,ni:ni+1], utils.improc.preprocess_color(rgb0_), valids=valids[0:1,:,ni:ni+1], cmap='spring', linewidth=2, frame_id=d_avg*100.0)


    sur_thr = 16
    dists = torch.norm(trajs_e/sc_pt - trajs_g/sc_pt, dim=-1) # B,S,N
    dist_ok = 1 - (dists > sur_thr).float() * valids # B,S,N
    survival = torch.cumprod(dist_ok, dim=1) # B,S,N
    metrics['survival'] = torch.mean(survival).item()
    
    # lifespans = np.ones((N))*(S-1)
    
    # for ni in range(N):
    #     done = False
    #     si = 1
    #     while (not done) and (si<S):
    #         if valids[0,si,ni] > 0:
    #             if dists[0,si,ni] > sur_thr:
    #                 lifespans[ni] = si-1
    #                 done = True
    #         si += 1
    # # print('lifespans', lifespans, 'S-1', S-1)
    # metrics['survival'] = np.mean(lifespans/(S-1))

    return metrics




# def test_model(model, d, device, dname='init', iters=8, sw=None, is_train=True, cheap_vis=False):
#     metrics = {}

#     rgbs = d['rgbs'].cuda().float() # B,S,C,H,W
#     trajs_g = d['trajs'].cuda().float() # B,S,N,2
#     valids = d['valids'].cuda().float() # B,S,N
#     if 'visibs' in metrics:
#         print('using real visibs')
#         vis_g = d['visibs'].cuda().float() # B,S,N
#     else:
#         vis_g = valids.clone()

#     B, S, C, H, W = rgbs.shape
#     B, S, N, D = trajs_g.shape
#     assert(D==2)
#     assert(B==1)

#     trajs_e0 = trajs_g[:,0:1].repeat(1,S,1,1)

#     preds, preds_anim, _ = model(trajs_e0, rgbs, iters=iters)
#     trajs_e = preds[-1]

#     l1_dists = torch.abs(trajs_e - trajs_g).sum(dim=-1) # B,S,N
#     l1_loss = utils.basic.reduce_masked_mean(l1_dists, valids)
#     l1_vis = utils.basic.reduce_masked_mean(l1_dists, valids*vis_g)
    
#     ate = torch.norm(trajs_e - trajs_g, dim=-1) # B,S,N
#     ate_all = utils.basic.reduce_masked_mean(ate, valids, dim=[1,2])
#     ate_vis = utils.basic.reduce_masked_mean(ate, valids*vis_g)
#     ate_occ = utils.basic.reduce_masked_mean(ate, valids*(1.0-vis_g))


#     metrics['l1'] = l1_loss.mean().item()
#     metrics['l1_vis'] = l1_vis.mean().item()
#     metrics['ate_all'] = ate_all.mean().item()
#     metrics['ate_vis'] = ate_vis.item()
#     metrics['ate_occ'] = ate_occ.item()
    
#     if sw is not None and sw.save_this:

#         prep_rgbs = utils.improc.preprocess_color(rgbs)
#         gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)

#         gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('', trajs_g[0:1], gray_rgbs[0:1].mean(dim=1), valids=valids[0:1], cmap='winter', only_return=True))

#         rgb_vis = []
#         for tre in preds_anim:
#             # tre = tre[:,:S]
#             ate = torch.norm(tre - trajs_g, dim=-1) # B,S,N
#             ate_all = utils.basic.reduce_masked_mean(ate, valids, dim=[1,2]) # B
#             rgb_vis.append(sw.summ_traj2ds_on_rgb('', tre[0:1], gt_rgb, valids=valids[0:1], only_return=True, cmap='spring', frame_id=ate_all[0]))
#             # rgb_vis.append(sw.summ_traj2ds_on_rgb('', tre[0:1], gt_rgb, only_return=True, cmap='spring'))
#         sw.summ_rgbs('3_test/animated_trajs_on_rgb', rgb_vis)
        
#     d_sum = 0.0
#     thrs = [1,2,4,8,16]
#     sx_ = W / 256.0
#     sy_ = H / 256.0
#     sc_py = np.array([sx_, sy_]).reshape([1,1,2])
#     sc_pt = torch.from_numpy(sc_py).float().cuda()
#     for thr in thrs:
#         # note we exclude timestep0 from this eval
#         d_ = (torch.norm(trajs_e[:,1:]/sc_pt - trajs_g[:,1:]/sc_pt, dim=-1) < thr).float() # B,S-1,N
#         d_ = utils.basic.reduce_masked_mean(d_, valids[:,1:]).item()
#         d_sum += d_
#         metrics['d_%d' % thr] = d_
#     d_avg = d_sum / len(thrs)
#     metrics['d_avg'] = d_avg

#     dists = torch.norm(trajs_e/sc_pt - trajs_g/sc_pt, dim=-1) # B,S,N
#     lifespans = np.ones((N))*(S-1)
#     for ni in range(N):
#         done = False
#         si = 0
#         while (not done) and (si<S):
#             if valids[0,si,ni] > 0:
#                 if dists[0,si,ni] > 16:
#                     lifespans[ni] = si-1
#                     done = True
#             si += 1
#     metrics['survival'] = np.mean(lifespans/(S-1))

#     return metrics



# def prep_sample(d, horz_flip=False, vert_flip=False, time_flip=False): 
#     # inflate batch size by flipping,
#     # and init some estimates, if we don't have them yet

#     rgbs = d['rgbs'].float() # B,S,C,H,W
#     track_g = d['track_g'].float() # B,S,N,8
#     trajs_g = track_g[:,:,:,:2]
#     vis_g = track_g[:,:,:,2]
#     valids = track_g[:,:,:,3]

#     B, S, C, H, W = rgbs.shape
#     assert(C==3)
#     N = trajs_g.shape[2]

#     use_tricks = False
#     if use_tricks:
#         if B==1:
#             # discard the trajs added purely for batching
#             trajs_g_ = trajs_g.reshape(S,N,2)
#             vis_g_ = vis_g.reshape(S,N)
#             valids_ = valids.reshape(S,N)

#             inds = valids_[0] > 0 # N
#             trajs_g_ = trajs_g_[:,inds]
#             vis_g_ = vis_g_[:,inds]
#             valids_ = valids_[:,inds]

#             N = trajs_g_.shape[1]
#             trajs_g = trajs_g_.reshape(1,S,N,2)
#             vis_g = vis_g_.reshape(1,S,N)
#             valids = valids_.reshape(1,S,N)

#         # pad = 32
#         # rgbs_ = rgbs.reshape(B*S, 3, H, W)
#         # # sometimes pad left or right; never both
#         # if np.random.rand() < 0.5: # left
#         #     rgbs_ = F.pad(rgbs_, (pad, 0, 0, 0), 'constant', 0)
#         #     trajs_g = trajs_g + np.array([1,0])*pad
#         #     W += pad
#         # else:
#         #     if np.random.rand() < 0.5: # right
#         #         rgbs_ = F.pad(rgbs_, (0, pad, 0, 0), 'constant', 0)
#         #         W += pad
#         # if np.random.rand() < 0.5: # top
#         #     rgbs_ = F.pad(rgbs_, (0, 0, pad, 0), 'constant', 0)
#         #     trajs_g = trajs_g + np.array([0,1])*pad
#         #     H += pad
#         # else:
#         #     if np.random.rand() < 0.5: # bottom
#         #         rgbs_ = F.pad(rgbs_, (0, 0, 0, pad), 'constant', 0)
#         #         H += pad
#         # _, _, H, W = rgbs_.shape
#         # rgbs = rgbs_.reshape(B,S,3,H,W)

#         if S > 8: 
#             # drop a random amount from the seq, to not overfit to a certain seqlen
#             S = S - np.random.randint(0, 4)
#             rgbs = rgbs[:,:S]
#             trajs_g = trajs_g[:,:S]
#             vis_g = vis_g[:,:S]
#             valids = valids[:,:S]
            
#         # print('rgbs', rgbs.shape)

#     # # random padding, so that the model does not always train in the same resolution
#     # pad_max = 16
#     # rgbs_ = rgbs.reshape(B*S, 3, H, W)
    
#     # pad = np.random.randint(1, pad_max)
#     # rgbs_ = F.pad(rgbs_, (pad, 0, 0, 0), 'constant', 0)
#     # trajs_g = trajs_g + np.array([1,0])*pad
#     # W += pad
    
#     # pad = np.random.randint(1, pad_max)
#     # rgbs_ = F.pad(rgbs_, (0, pad, 0, 0), 'constant', 0)
#     # W += pad
    
#     # pad = np.random.randint(1, pad_max)
#     # rgbs_ = F.pad(rgbs_, (0, 0, pad, 0), 'constant', 0)
#     # trajs_g = trajs_g + np.array([0,1])*pad
#     # H += pad
    
#     # pad = np.random.randint(1, pad_max)
#     # rgbs_ = F.pad(rgbs_, (0, 0, 0, pad), 'constant', 0)
#     # H += pad
    
#     # # now make it divisible by 32
#     # div = 32
#     # pad_ht = (((H // div) + 1) * div - H) % div
#     # pad_wd = (((W // div) + 1) * div - W) % div
#     # if np.random.rand() < 0.5: 
#     #     rgbs_ = F.pad(rgbs_, (pad_wd//2, pad_wd-pad_wd//2, pad_ht//2, pad_ht-pad_ht//2), 'constant', 0)
#     #     trajs_g = trajs_g + np.array([1,0])*pad_wd//2
#     #     trajs_g = trajs_g + np.array([0,1])*pad_ht//2
#     # else:
#     #     rgbs_ = F.pad(rgbs_, (0, pad_wd, 0, pad_ht), 'constant', 0)
#     # _, _, H, W = rgbs_.shape
#     # rgbs = rgbs_.reshape(B,S,3,H,W)
        
    
#     if horz_flip: # increase the batchsize by horizontal flipping
#         rgbs_flip = torch.flip(rgbs, [-1])
#         trajs_g_flip = trajs_g.clone()
#         trajs_g_flip[:,:,:,0] = W-1 - trajs_g_flip[:,:,:,0]
#         vis_g_flip = vis_g.clone()
#         valids_flip = valids.clone()
#         trajs_g = torch.cat([trajs_g, trajs_g_flip], dim=0)
#         vis_g = torch.cat([vis_g, vis_g_flip], dim=0)
#         valids = torch.cat([valids, valids_flip], dim=0)
#         rgbs = torch.cat([rgbs, rgbs_flip], dim=0)
#         B = B * 2

#     if vert_flip: # increase the batchsize by vertical flipping
#         rgbs_flip = torch.flip(rgbs, [-2])
#         trajs_g_flip = trajs_g.clone()
#         trajs_g_flip[:,:,:,1] = H-1 - trajs_g_flip[:,:,:,1]
#         vis_g_flip = vis_g.clone()
#         valids_flip = valids.clone()
#         trajs_g = torch.cat([trajs_g, trajs_g_flip], dim=0)
#         vis_g = torch.cat([vis_g, vis_g_flip], dim=0)
#         valids = torch.cat([valids, valids_flip], dim=0)
#         rgbs = torch.cat([rgbs, rgbs_flip], dim=0)
#         B = B * 2

#     if time_flip: # increase the batchsize by temporal flipping/shuffling

#         perm = np.random.permutation(S-1)+1
#         perm = np.concatenate([[0], perm], axis=0)

#         rgbs_flip = rgbs[:,perm]
#         trajs_g_flip = trajs_g[:,perm]
#         vis_g_flip = vis_g[:,perm]
#         valids_flip = valids[:,perm]

#         trajs_g = torch.cat([trajs_g, trajs_g_flip], dim=0)
#         vis_g = torch.cat([vis_g, vis_g_flip], dim=0)
#         valids = torch.cat([valids, valids_flip], dim=0)
#         rgbs = torch.cat([rgbs, rgbs_flip], dim=0)
#         B = B * 2
#     # print('arrived at BS=%d' % (B*S))

#     # with low prob, shuffle the time axis,
#     # making flow an unreliable signal,
#     # and forcing each timestep to fight for himself
#     # we do this after inflation, so that each batch contains a mix
#     for b in range(B):
#         if np.random.rand() < 0.2: 
#             perm = np.random.permutation(S-1)+1
#             perm = np.concatenate([[0], perm], axis=0)
#             # print('perm', perm)
#             rgbs[b] = rgbs[b,perm]
#             trajs_g[b] = trajs_g[b,perm]
#             vis_g[b] = vis_g[b,perm]
#             valids[b] = valids[b,perm]

#     # # init using offset from gt
#     # x = torch.from_numpy(np.random.uniform(-W//4, W//4, (B,S,N))).float().to(trajs_g.device)
#     # y = torch.from_numpy(np.random.uniform(-H//4, H//4, (B,S,N))).float().to(trajs_g.device)
#     # offsets = torch.stack([x,y], dim=-1) # B,S,N,2
#     # trajs_e0 = trajs_g + offsets

#     # if np.random.rand() < 0.5:
#     #     # init closer to gt
#     #     trajs_e0 = (trajs_e0 + trajs_g)/2.0

#     # full random
#     x = torch.from_numpy(np.random.uniform(0, W-1, (B,S,N))).float().to(trajs_g.device)
#     y = torch.from_numpy(np.random.uniform(0, H-1, (B,S,N))).float().to(trajs_g.device)
#     trajs_e0 = torch.stack([x,y], dim=-1) # B,S,N,2

#     # # init using offset from gt
#     # x = torch.from_numpy(np.random.uniform(-W//4, W//4, (B,S,N))).float().to(trajs_g.device)
#     # y = torch.from_numpy(np.random.uniform(-H//4, H//4, (B,S,N))).float().to(trajs_g.device)
#     # offsets = torch.stack([x,y], dim=-1) # B,S,N,2
#     # trajs_e0 = trajs_g + offsets

    
#     # # move halfway to zeroth step
#     # trajs_e0 = (trajs_e0 + trajs_g[:,0:1])/2.0

#     # if np.random.rand() < 0.5:
#     #     # init closer to gt
#     #     trajs_e0 = (trajs_e0 + trajs_g)/2.0

#     # if np.random.rand() < 0.5:
#     #     # init closer to gt
#     #     trajs_e0 = (trajs_e0 + trajs_g)/2.0

#     # coeff = np.random.rand()

#     coeff = torch.from_numpy(np.random.uniform(0, 1, (B,1,N,2))).float().to(trajs_g.device)
#     trajs_e0 = trajs_e0*coeff + trajs_g*(1-coeff)
    
#     # reset zeroth, so that it's clear what to track
#     trajs_e0[:,0] = trajs_g[:,0]

#     # # init zero vel
#     # trajs_e0 = trajs_g[:,0:1].repeat(1,S,1,1)


#     new_d = {}
#     new_d['rgbs'] = rgbs
#     new_d['trajs_g'] = trajs_g
#     new_d['vis_g'] = vis_g
#     new_d['valids'] = valids
#     new_d['trajs_e0'] = trajs_e0
#     new_d['step'] = 0
#     # new_d['ate_prev'] = utils.basic.reduce_masked_mean(torch.norm(trajs_e0 - trajs_g, dim=-1), valids).item()
#     # new_d['ate_now'] = utils.basic.reduce_masked_mean(torch.norm(trajs_e0 - trajs_g, dim=-1), valids).item()

#     return new_d
        
    
def run_model(model, d, device, dname='init', iters=8, sw=None, is_train=True, cheap_vis=False):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    metrics = {}

    rgbs = d['rgbs'].float().to(device) # B,S,C,H,W
    track_g = d['track_g'].float().to(device) # B,S,N,8

    # print('rgbs', rgbs.shape)
    # print('track_g', track_g.shape)
    
    trajs_g = track_g[:,:,:,:2]
    vis_g = track_g[:,:,:,2]
    valids = track_g[:,:,:,3]

    if np.random.rand() < 0.5:
        rgbs = rgbs.permute(0,1,2,4,3) # swap xy
        trajs_g = trajs_g.clone().flip([3]) # swap xy
        
    # print('rgbs', rgbs.shape)
    # # print('trajs_e0', trajs_e0.shape)
    # print('trajs_g', trajs_g.shape)
    # print('vis_g', vis_g.shape)
    # print('valids', valids.shape, torch.sum(valids[:,0]))

    B, S, C, H, W = rgbs.shape
    assert(C==3)
    B, S, N, D = trajs_g.shape
    assert(D==2)

    # full random
    x = torch.from_numpy(np.random.uniform(0, W-1, (B,S,N))).float().to(trajs_g.device)
    y = torch.from_numpy(np.random.uniform(0, H-1, (B,S,N))).float().to(trajs_g.device)
    trajs_e0 = torch.stack([x,y], dim=-1) # B,S,N,2
    # mix a random amount with gt
    coeff = torch.from_numpy(np.random.uniform(0, 1, (B,1,N,1))).float().to(trajs_g.device)
    trajs_e0 = trajs_e0*coeff + trajs_g*(1-coeff)

    # use zero-velocity init for some trajs
    trajs_z = trajs_g[:,0:1].repeat(1,S,1,1)
    mask = (torch.from_numpy(np.random.uniform(0, 1, (B,1,N,1))).float().to(trajs_g.device)>0.1).float()
    trajs_e0 = trajs_e0*mask + trajs_z*(1-mask)
        
    # reset zeroth
    trajs_e0[:,0:1] = trajs_g[:,0:1]
    
    # # add some noise, to ensure every element of the batch is unique
    # offsets = torch.from_numpy(np.random.uniform(-4, 4, (B,S,N,2))).float().to(trajs_g.device)
    # trajs_e0 = trajs_e0 + offsets
    # trajs_e0[:,0] = trajs_g[:,0]

    ate0 = torch.norm(trajs_e0 - trajs_g, dim=-1) # B,S,N
    ate0_all = utils.basic.reduce_masked_mean(ate0, valids, dim=[1,2])
    
    # if sw is not None and sw.save_this:
    #     # rgb0 = utils.improc.preprocess_color(rgbs[0:1,0])
    #     # sw.summ_traj2ds_on_rgb('0_inputs/trajs_g_on_rgb0', trajs_g[0:1], rgb0, cmap='winter', linewidth=2)
        
    #     sw.summ_traj2ds_on_rgb('%s_0_inputs/trajs_g_on_rgb' % dname, trajs_g[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1), valids=valids[0:1], cmap='winter')
    #     sw.summ_traj2ds_on_rgb('%s_0_inputs/trajs_e0_on_rgb' % dname, trajs_e0[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1), valids=valids[0:1], cmap='spring', frame_id=ate0_all[0].item())
    #     # sw.summ_traj2ds_on_rgbs2('%s_0_inputs/trajs_g_on_rgbs2' % dname, trajs_g[0:1], vis_g[0:1], utils.improc.preprocess_color(rgbs[0:1]), valids=valids[0:1])

    # trajs_e, l1_loss, l1_loss_early, ce_loss = model(trajs_e0, rgbs, trajs_g=trajs_g, vis_g=vis_g, valids=valids, sw=sw)
    # trajs_e = model(trajs_e0, rgbs, sw=sw)

    preds, preds_anim, loss = model(trajs_e0, rgbs, iters=iters, trajs_g=trajs_g, vis_g=vis_g, valids=valids)
    trajs_e = preds[-1]

    # print('loss', loss)
    # total_loss += l1_loss.mean()
    # total_loss += l1_loss_early.mean()*0.1
    # total_loss += ce_loss.mean()

    # print('trajs_e', trajs_e.shape)
    # print('trajs_g', trajs_g.shape)
    # print('preds_anim[-1]', preds_anim[-1].shape)

    # preds_ = torch.stack(preds, dim=0)[1:] # I,B,S,N,2; # exclude zeroth, since that's the init
    # trajs_g_ = trajs_g.unsqueeze(0).repeat(iters,1,1,1,1)
    # valids_ = valids.unsqueeze(0).repeat(iters,1,1,1)
    # l1_dists_ = torch.abs(trajs_e - trajs_g_).sum(dim=-1) # I,B,S,N
    # l1_loss_ = utils.basic.reduce_masked_mean(l1_dists_, valids_) 
    # total_loss += l1_loss_

    total_loss += loss
    
    # for tre in preds_anim:
    #     l1_dists = torch.abs(tre - trajs_g).sum(dim=-1) # B,S,N
    #     l1_loss = utils.basic.reduce_masked_mean(l1_dists, valids)
    #     total_loss += l1_loss.mean()/float(iters)

    l1_dists = torch.abs(trajs_e - trajs_g).sum(dim=-1) # B,S,N
    l1_loss = utils.basic.reduce_masked_mean(l1_dists, valids)
    l1_vis = utils.basic.reduce_masked_mean(l1_dists, valids*vis_g)
    # total_loss += l1_loss.mean()
    
    ate = torch.norm(trajs_e - trajs_g, dim=-1) # B,S,N
    ate_all = utils.basic.reduce_masked_mean(ate, valids, dim=[1,2])
    ate_vis = utils.basic.reduce_masked_mean(ate, valids*vis_g)
    ate_occ = utils.basic.reduce_masked_mean(ate, valids*(1.0-vis_g))

    d_sum = 0.0
    thrs = [1,2,4,8,16]
    sx_ = W / 256.0
    sy_ = H / 256.0
    sc_py = np.array([sx_, sy_]).reshape([1,1,2])
    sc_pt = torch.from_numpy(sc_py).float().cuda()
    for thr in thrs:
        # note we exclude timestep0 from this eval
        d_ = (torch.norm(trajs_e[:,1:]/sc_pt - trajs_g[:,1:]/sc_pt, dim=-1) < thr).float() # B,S-1,N
        d_ = utils.basic.reduce_masked_mean(d_, valids[:,1:]).item()
        d_sum += d_
        metrics['d_%d' % thr] = d_
    d_avg = d_sum / len(thrs)
    metrics['d_avg'] = d_avg

    metrics['l1'] = l1_loss.mean().item()
    # metrics['l1_early'] = l1_loss_early.mean().item()
    # metrics['ce'] = ce_loss.mean().item()
    metrics['l1_vis'] = l1_vis.mean().item()
    metrics['ate_all'] = ate_all.mean().item()
    metrics['ate_vis'] = ate_vis.item()
    metrics['ate_occ'] = ate_occ.item()
    metrics['total_loss'] = total_loss.item()

    if sw is not None and sw.save_this:
        
        # rgb0 = utils.improc.preprocess_color(rgbs[0:1,0])
        # sw.summ_traj2ds_on_rgb('0_inputs/trajs_g_on_rgb0', trajs_g[0:1], rgb0, cmap='winter', linewidth=2)

        prep_rgbs = utils.improc.preprocess_color(rgbs)
        gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
        

        rgb0 = sw.summ_traj2ds_on_rgb('', trajs_g[0:1], prep_rgbs[0:1,0], valids=valids[0:1], cmap='winter', linewidth=2, only_return=True)
        sw.summ_traj2ds_on_rgb('0_inputs/%s_trajs_e0_on_rgb0' % dname, trajs_e0[0:1], utils.improc.preprocess_color(rgb0), valids=valids[0:1], cmap='spring', linewidth=2, frame_id=ate0_all[0].mean().item())
        sw.summ_traj2ds_on_rgb('2_outputs/%s_trajs_e_on_rgb0' % dname, trajs_e[0:1], utils.improc.preprocess_color(rgb0), valids=valids[0:1], cmap='spring', linewidth=2, frame_id=ate_all[0].mean().item())
        # sw.summ_traj2ds_on_rgbs2('0_inputs/%s_trajs_g_on_rgbs2' % dname, trajs_g[0:1,::4], vis_g[0:1,::4], prep_rgbs[0:1,::4], valids=valids[0:1,::4], frame_ids=list(range(0,S,4)))
        
        # in the kp vis, clamp so that we can see everything
        trajs_g_clamp = trajs_g.clone()
        trajs_g_clamp[:,:,:,0] = trajs_g_clamp[:,:,:,0].clip(0,W-1)
        trajs_g_clamp[:,:,:,1] = trajs_g_clamp[:,:,:,1].clip(0,H-1)
        trajs_e_clamp = trajs_e.clone()
        trajs_e_clamp[:,:,:,0] = trajs_e_clamp[:,:,:,0].clip(0,W-1)
        trajs_e_clamp[:,:,:,1] = trajs_e_clamp[:,:,:,1].clip(0,H-1)

        gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('', trajs_g[0:1], gray_rgbs[0:1].mean(dim=1), valids=valids[0:1], cmap='winter', only_return=True))
        rgb_vis = []
        for tre in preds_anim:
            # tre = tre[:,:S]
            ate = torch.norm(tre - trajs_g, dim=-1) # B,S,N
            ate_all = utils.basic.reduce_masked_mean(ate, valids, dim=[1,2]) # B
            rgb_vis.append(sw.summ_traj2ds_on_rgb('', tre[0:1], gt_rgb, valids=valids[0:1], only_return=True, cmap='spring', frame_id=ate_all[0]))
            # rgb_vis.append(sw.summ_traj2ds_on_rgb('', tre[0:1], gt_rgb, only_return=True, cmap='spring'))
        sw.summ_rgbs('3_test/animated_trajs_on_rgb', rgb_vis)

        if False:
            outs = sw.summ_pts_on_rgbs(
                '',
                trajs_g_clamp[0:1,::4],
                gray_rgbs[0:1,::4],
                valids=valids[0:1,::4],
                cmap='winter', linewidth=3, only_return=True)
            sw.summ_pts_on_rgbs(
                '0_inputs/%s_kps_gv_on_rgbs' % dname,
                trajs_g_clamp[0:1,::4],
                utils.improc.preprocess_color(outs),
                valids=valids[0:1,::4],
                cmap='spring', linewidth=2)

            outs = sw.summ_pts_on_rgbs(
                '',
                trajs_g_clamp[0:1,::4],
                gray_rgbs[0:1,::4],
                valids=valids[0:1,::4],
                cmap='winter', linewidth=3, only_return=True)
            sw.summ_pts_on_rgbs(
                '2_outputs/%s_kps_eg_on_rgbs' % dname,
                trajs_e_clamp[0:1,::4],
                utils.improc.preprocess_color(outs),
                valids=valids[0:1,::4],
                cmap='spring', linewidth=2)
        

    # save our work, so that we can use it as init later
    trajs_e0 = trajs_e.detach()

    # reset zeroth, so that it's clear what to track
    trajs_e0[:,0] = trajs_g[:,0]

    # work = {}
    # work['rgbs'] = rgbs.cpu()
    # # work['ate_prev'] = sample['ate_now'] 
    # # work['ate_now'] = metrics['ate_vis']
    # work['trajs_e0'] = trajs_e0.cpu()
    # work['trajs_g'] = trajs_g.cpu()
    # work['vis_g'] = vis_g.cpu()
    # work['valids'] = valids.cpu()
    # work['step'] = d['step'] + 1
    
    return total_loss, metrics#, work
    

def main(
        exp_name='debug',
        B=1, # batchsize 
        S=120, # seqlen
        stride=8, # spatial stride of the model 
        iters=16, # inference steps of the model
        image_size=(512,896), # input resolution
        shuffle=False, # dataset shuffling
        log_freq=99,
        max_iters=30,
        log_dir='./logs_test_on_tap',
        dataset_location='/orion/u/aharley/datasets/tapvid_davis',
        init_dir='./reference_model',
        # cuda
        device_ids=[0],
        n_pool=1000,
):
    device = 'cuda:%d' % device_ids[0]

    # the idea in this file is:
    # load a ckpt, and test it in tapvid,
    # tracking points from frame0 to the end
    
    exp_name = 'tt00' # copy from dev repo
    exp_name = 'tt01' # clean up
    exp_name = 'tt02' # clean the net

    assert(image_size[0] % 32 == 0)
    assert(image_size[1] % 32 == 0)
    
    ## autogen a descriptive name
    model_name = "%d_%d" % (B,S)
    model_name += "_i%d" % (iters)
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H%M%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    writer_x = SummaryWriter(log_dir + '/' + model_name + '/x', max_queue=10, flush_secs=60)

    dataset_x = TapVidDavis(
        dataset_location=dataset_location,
    )
    dataloader_x = DataLoader(
        dataset_x,
        batch_size=1,
        shuffle=shuffle,
        num_workers=1)
    iterloader_x = iter(dataloader_x)

    model = Pips(stride=stride).to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    parameters = list(model.parameters())

    from prettytable import PrettyTable
    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            if param > 100000:
                table.add_row([name, param])
            total_params+=param
        print(table)
        print('total params: %.2f M' % (total_params/1000000.0))
        return total_params
    count_parameters(model)

    global_step = 0
    _ = saverloader.load(init_dir, model.module)
    requires_grad(parameters, False)
    model.eval()

    pools_x = create_pools(n_pool)

    while global_step < max_iters:
        global_step += 1
        iter_start_time = time.time()
        with torch.no_grad():
            torch.cuda.empty_cache()
        sw_x = utils.improc.Summ_writer(
            writer=writer_x,
            global_step=global_step,
            log_freq=log_freq,
            fps=min(S,8),
            scalar_freq=int(log_freq/4),
            just_gif=True)
        try:
            sample = next(iterloader_x)
        except StopIteration:
            iterloader_x = iter(dataloader_x)
            sample = next(iterloader_x)
        iter_rtime = time.time()-iter_start_time
        with torch.no_grad():
            metrics = test_on_fullseq(model, sample, sw_x, iters=iters, S_max=S, image_size=image_size)
        for key in list(pools_x.keys()):
            if key in metrics:
                pools_x[key].update([metrics[key]])
                sw_x.summ_scalar('_/%s' % (key), pools_x[key].mean())
        iter_itime = time.time()-iter_start_time
        
        print('%s; step %06d/%d; rtime %.2f; itime %.2f; d_x %.1f; sur_x %.1f' % (
            model_name, global_step, max_iters, iter_rtime, iter_itime,
            pools_x['d_avg'].mean()*100.0, pools_x['survival'].mean()*100.0))
            
    writer_x.close()
            

if __name__ == '__main__':
    Fire(main)
