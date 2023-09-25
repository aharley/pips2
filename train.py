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
from datasets.exportdataset import ExportDataset
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
    pool_names = [
        'l1',
        'd_1',
        'd_2',
        'd_4',
        'd_8',
        'd_16',
        'd_avg',
        'l1_vis',
        'ate_all',
        'ate_vis',
        'ate_occ',
        'median_l2',
        'survival',
        'total_loss',
    ]
    for pool_name in pool_names:
        pools[pool_name] = utils.misc.SimplePool(n_pool, version='np')
    return pools

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, lr, num_steps+100, pct_start=0.1, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler

def val_model(model, d, device, iters=8, sw=None, is_train=False):
    metrics = {}

    rgbs = d['rgbs'].float().to(device) # B,S,C,H,W
    track_g = d['track_g'].float().to(device) # B,S,N,8
    trajs_g = track_g[:,:,:,:2]
    vis_g = track_g[:,:,:,2]
    valids = track_g[:,:,:,3]

    B, S, C, H, W = rgbs.shape
    B, S, N, D = trajs_g.shape
    assert(D==2)

    # zero-vel init
    trajs_e0 = trajs_g[:,0:1].repeat(1,S,1,1)

    preds, preds_anim, _, _ = model(trajs_e0, rgbs, iters=iters)
    trajs_e = preds[-1]

    l1_dists = torch.abs(trajs_e - trajs_g).sum(dim=-1) # B,S,N
    l1_loss = utils.basic.reduce_masked_mean(l1_dists, valids)
    l1_vis = utils.basic.reduce_masked_mean(l1_dists, valids*vis_g)
    
    ate = torch.norm(trajs_e - trajs_g, dim=-1) # B,S,N
    ate_all = utils.basic.reduce_masked_mean(ate, valids, dim=[1,2])
    ate_vis = utils.basic.reduce_masked_mean(ate, valids*vis_g)
    ate_occ = utils.basic.reduce_masked_mean(ate, valids*(1.0-vis_g))

    metrics['l1'] = l1_loss.mean().item()
    metrics['l1_vis'] = l1_vis.mean().item()
    metrics['ate_all'] = ate_all.mean().item()
    metrics['ate_vis'] = ate_vis.item()
    metrics['ate_occ'] = ate_occ.item()
    
    if sw is not None and sw.save_this:
        prep_rgbs = utils.improc.preprocess_color(rgbs)
        prep_grays = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
        gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('', trajs_g[0:1], prep_grays[0:1].mean(dim=1), valids=valids[0:1], cmap='winter', only_return=True))
        rgb_vis = []
        for tre in preds_anim:
            ate = torch.norm(tre - trajs_g, dim=-1) # B,S,N
            ate_all = utils.basic.reduce_masked_mean(ate, valids, dim=[1,2]) # B
            rgb_vis.append(sw.summ_traj2ds_on_rgb('', tre[0:1], gt_rgb, valids=valids[0:1], only_return=True, cmap='spring', frame_id=ate_all[0]))
        sw.summ_rgbs('3_test/animated_trajs_on_rgb', rgb_vis)
        
    d_sum = 0.0
    thrs = [1,2,4,8,16]
    sx_ = W / 256.0
    sy_ = H / 256.0
    sc_py = np.array([sx_, sy_]).reshape([1,1,2])
    sc_pt = torch.from_numpy(sc_py).float().cuda()
    for thr in thrs:
        # note we exclude timestep0 from this eval
        d_ = (torch.norm(trajs_e[:,1:]/sc_pt - trajs_g[:,1:]/sc_pt, dim=-1) < thr).float() # B,S-1,N
        d_ = utils.basic.reduce_masked_mean(d_, valids[:,1:]).item()*100.0
        d_sum += d_
        metrics['d_%d' % thr] = d_
    d_avg = d_sum / len(thrs)
    metrics['d_avg'] = d_avg

    sur_thr = 16
    dists = torch.norm(trajs_e/sc_pt - trajs_g/sc_pt, dim=-1) # B,S,N
    dist_ok = 1 - (dists > sur_thr).float() * valids # B,S,N
    survival = torch.cumprod(dist_ok, dim=1) # B,S,N
    metrics['survival'] = torch.mean(survival).item()*100.0
    
    # get the median l2 error for each trajectory
    dists_ = dists.permute(0,2,1).reshape(B*N,S)
    valids_ = valids.permute(0,2,1).reshape(B*N,S)
    median_l2 = utils.basic.reduce_masked_median(dists_, valids_, keep_batch=True) # B*N
    metrics['median_l2'] = median_l2.mean().item()

    return metrics

def run_model(model, d, device, iters=8, sw=None, is_train=True, use_augs=True):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    metrics = {}

    rgbs = d['rgbs'].float().to(device) # B,S,C,H,W
    track_g = d['track_g'].float().to(device) # B,S,N,8
    trajs_g = track_g[:,:,:,:2]
    vis_g = track_g[:,:,:,2]
    valids = track_g[:,:,:,3]

    if use_augs and np.random.rand() < 0.5: # rot90 aug
        rgbs = rgbs.permute(0,1,2,4,3) # swap xy
        trajs_g = trajs_g.flip([3]) # swap xy
        
    B, S, C, H, W = rgbs.shape
    assert(C==3)
    B, S, N, D = trajs_g.shape
    assert(D==2)

    # full random
    x = torch.from_numpy(np.random.uniform(0, W-1, (B,S,N))).float().to(trajs_g.device)
    y = torch.from_numpy(np.random.uniform(0, H-1, (B,S,N))).float().to(trajs_g.device)
    trajs_e0 = torch.stack([x,y], dim=-1) # B,S,N,2
    # mix with gt a random amount
    coeff = torch.from_numpy(np.random.uniform(0, 1, (B,1,N,1))).float().to(trajs_g.device)
    trajs_e0 = trajs_e0*coeff + trajs_g*(1-coeff)
    # use zero-velocity init for some
    trajs_z = trajs_g[:,0:1].repeat(1,S,1,1)
    mask = (torch.from_numpy(np.random.uniform(0, 1, (B,1,N,1))).float().to(trajs_g.device)>0.5).float()
    trajs_e0 = trajs_e0*mask + trajs_z*(1-mask)
    # reset zeroth on all
    trajs_e0[:,0:1] = trajs_g[:,0:1]

    # measure our initial distance, so we can check our improvement
    ate0 = torch.norm(trajs_e0 - trajs_g, dim=-1) # B,S,N
    ate0_all = utils.basic.reduce_masked_mean(ate0, valids, dim=[1,2])
    
    preds, preds_anim, _, loss = model(trajs_e0, rgbs, iters=iters, trajs_g=trajs_g, vis_g=vis_g, valids=valids, is_train=is_train)
    trajs_e = preds[-1]

    total_loss += loss

    # collect stats
    l1_dists = torch.abs(trajs_e - trajs_g).sum(dim=-1) # B,S,N
    l1_loss = utils.basic.reduce_masked_mean(l1_dists, valids)
    l1_vis = utils.basic.reduce_masked_mean(l1_dists, valids*vis_g)
    ate = torch.norm(trajs_e - trajs_g, dim=-1) # B,S,N
    ate_all = utils.basic.reduce_masked_mean(ate, valids, dim=[1,2])
    ate_vis = utils.basic.reduce_masked_mean(ate, valids*vis_g)
    ate_occ = utils.basic.reduce_masked_mean(ate, valids*(1.0-vis_g))
    metrics['l1'] = l1_loss.mean().item()
    metrics['l1_vis'] = l1_vis.mean().item()
    metrics['ate_all'] = ate_all.mean().item()
    metrics['ate_vis'] = ate_vis.item()
    metrics['ate_occ'] = ate_occ.item()
    metrics['total_loss'] = total_loss.item()

    d_sum = 0.0
    thrs = [1,2,4,8,16]
    sx_ = W / 256.0
    sy_ = H / 256.0
    sc_py = np.array([sx_, sy_]).reshape([1,1,2])
    sc_pt = torch.from_numpy(sc_py).float().to(device)
    for thr in thrs:
        # note we exclude timestep0 from this eval
        d_ = (torch.norm(trajs_e[:,1:]/sc_pt - trajs_g[:,1:]/sc_pt, dim=-1) < thr).float() # B,S-1,N
        d_ = utils.basic.reduce_masked_mean(d_, valids[:,1:]).item()*100.0
        d_sum += d_
        metrics['d_%d' % thr] = d_
    d_avg = d_sum / len(thrs)
    metrics['d_avg'] = d_avg

    sur_thr = 16
    dists = torch.norm(trajs_e/sc_pt - trajs_g/sc_pt, dim=-1) # B,S,N
    dist_ok = 1 - (dists > sur_thr).float() * valids # B,S,N
    survival = torch.cumprod(dist_ok, dim=1) # B,S,N
    metrics['survival'] = torch.mean(survival).item()*100.0
    
    # get the median l2 error for each trajectory
    dists_ = dists.permute(0,2,1).reshape(B*N,S)
    valids_ = valids.permute(0,2,1).reshape(B*N,S)
    val_ok = valids_[:,0] > 0 # get rid of the ones we padded in
    dists_ = dists_[val_ok]
    valids_ = valids_[val_ok]
    median_l2 = utils.basic.reduce_masked_median(dists_, valids_, keep_batch=True) # B*N
    metrics['median_l2'] = median_l2.mean().item()

    if sw is not None and sw.save_this:
        prep_rgbs = utils.improc.preprocess_color(rgbs)
        prep_grays = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)

        rgb0 = sw.summ_traj2ds_on_rgb('', trajs_g[0:1], prep_rgbs[0:1,0], valids=valids[0:1], cmap='winter', linewidth=2, only_return=True)
        sw.summ_traj2ds_on_rgb('0_inputs/trajs_e0_on_rgb0', trajs_e0[0:1], utils.improc.preprocess_color(rgb0), valids=valids[0:1], cmap='spring', linewidth=2, frame_id=ate0_all[0].mean().item())
        sw.summ_traj2ds_on_rgb('2_outputs/trajs_e_on_rgb0', trajs_e[0:1], utils.improc.preprocess_color(rgb0), valids=valids[0:1], cmap='spring', linewidth=2, frame_id=ate_all[0].mean().item())
        sw.summ_traj2ds_on_rgbs2('0_inputs/trajs_g_on_rgbs2', trajs_g[0:1,::4], vis_g[0:1,::4], prep_rgbs[0:1,::4], valids=valids[0:1,::4], frame_ids=list(range(0,S,4)))
        
        # in the kp vis, clamp so that we can see everything
        trajs_g_clamp = trajs_g.clone()
        trajs_g_clamp[:,:,:,0] = trajs_g_clamp[:,:,:,0].clip(0,W-1)
        trajs_g_clamp[:,:,:,1] = trajs_g_clamp[:,:,:,1].clip(0,H-1)
        trajs_e_clamp = trajs_e.clone()
        trajs_e_clamp[:,:,:,0] = trajs_e_clamp[:,:,:,0].clip(0,W-1)
        trajs_e_clamp[:,:,:,1] = trajs_e_clamp[:,:,:,1].clip(0,H-1)

        gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('', trajs_g[0:1], prep_grays[0:1].mean(dim=1), valids=valids[0:1], cmap='winter', only_return=True))
        rgb_vis = []
        for tre in preds_anim:
            ate = torch.norm(tre - trajs_g, dim=-1) # B,S,N
            ate_all = utils.basic.reduce_masked_mean(ate, valids, dim=[1,2]) # B
            rgb_vis.append(sw.summ_traj2ds_on_rgb('', tre[0:1], gt_rgb, valids=valids[0:1], only_return=True, cmap='spring', frame_id=ate_all[0]))
        sw.summ_rgbs('3_test/animated_trajs_on_rgb', rgb_vis)

        outs = sw.summ_pts_on_rgbs(
            '',
            trajs_g_clamp[0:1,::4],
            prep_grays[0:1,::4],
            valids=valids[0:1,::4],
            cmap='winter', linewidth=3, only_return=True)
        sw.summ_pts_on_rgbs(
            '0_inputs/kps_gv_on_rgbs',
            trajs_g_clamp[0:1,::4],
            utils.improc.preprocess_color(outs),
            valids=valids[0:1,::4]*vis_g[0:1,::4],
            cmap='spring', linewidth=2)

        outs = sw.summ_pts_on_rgbs(
            '',
            trajs_g_clamp[0:1,::4],
            prep_grays[0:1,::4],
            valids=valids[0:1,::4],
            cmap='winter', linewidth=3, only_return=True)
        sw.summ_pts_on_rgbs(
            '2_outputs/kps_eg_on_rgbs',
            trajs_e_clamp[0:1,::4],
            utils.improc.preprocess_color(outs),
            valids=valids[0:1,::4],
            cmap='spring', linewidth=2)
        
    return total_loss, metrics
    

def main(
        B=2, # batchsize 
        S=36, # seqlen
        N=64, # number of points per clip
        stride=8, # spatial stride of the model 
        iters=6, # inference steps of the model
        crop_size=(384,512), # raw flt data is 540,960
        use_augs=True, # resizing/jittering/color/blur augs
        shuffle=True, # dataset shuffling
        cache_len=0, # how many samples to cache into ram (for overfitting/debug)
        cache_freq=0, # how often to add a new sample to cache
        dataset_location='./pod_export', # where we exported the data
        dataset_version='ae_36_128_384x512', # export version
        n_pool=1000, # size of running avg for stats
        quick=False, # debug
        # optimization
        lr=5e-4,
        grad_acc=1, 
        use_scheduler=True,
        max_iters=200000,
        # summaries
        log_dir='./logs_train',
        log_freq=1000,
        val_freq=100,
        # saving/loading
        ckpt_dir='./checkpoints',
        save_freq=1000,
        keep_latest=2,
        init_dir='',
        load_optimizer=True,
        load_step=True,
        ignore_load=None,
        device_ids=[0],
):
    device = 'cuda:%d' % device_ids[0]

    # the idea in this file is:
    # train from scratch on pointodyssey.
    # on val steps, unroll the inference.

    exp_name = 'aa00' # copy from dev repo
    exp_name = 'aa01' # clean up

    if quick: # (debug)
        B = 1
        log_freq = 100
        max_iters = 1000
        shuffle = False
        val_freq = 10
        n_pool = 100
        use_augs = False
        cache_len = 0 # overfit on this many
        cache_freq = 0
        save_freq = 99999999
        
    if init_dir:
        init_dir = '%s/%s' % (ckpt_dir, init_dir)
        
    assert(crop_size[0] % 32 == 0)
    assert(crop_size[1] % 32 == 0)
    
    # autogen a descriptive name
    model_name = "%d_%d_%d" % (B,S,N)
    model_name += "_i%d" % (iters)
    if grad_acc > 1:
        model_name += "x%d" % grad_acc
    lrn = "%.1e" % lr # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1] # e.g., 5e-4
    model_name += "_%s" % lrn
    if use_scheduler:
        model_name += "s"
    if cache_len:
        model_name += "_c%d_f%d" % (cache_len, cache_freq)
    if use_augs:
        model_name += "_A"
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H%M%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    ckpt_path = '%s/%s' % (ckpt_dir, model_name)
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)
    if val_freq:
        writer_v = SummaryWriter(log_dir + '/' + model_name + '/v', max_queue=10, flush_secs=60)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    dataset_t = ExportDataset(
        dataset_location=dataset_location,
        dataset_version=dataset_version,
        S=S,
        crop_size=crop_size,
        use_augs=use_augs)
    dataloader_t = DataLoader(
        dataset_t,
        batch_size=B,
        shuffle=shuffle,
        num_workers=6,
        worker_init_fn=worker_init_fn,
        drop_last=True)
    iterloader_t = iter(dataloader_t)

    if cache_len:
        sample_pool_t = utils.misc.SimplePool(cache_len, version='np')
    
    model = Pips(stride=stride).to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    parameters = list(model.parameters())
    weight_decay = 1e-6
    if use_scheduler:
        optimizer, scheduler = fetch_optimizer(lr, weight_decay, 1e-8, max_iters, model.parameters())
    else:
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay) 
        scheduler = None

    utils.misc.count_parameters(model)

    global_step = 0
    if init_dir:
        if load_step and load_optimizer:
            global_step = saverloader.load(init_dir, model.module, optimizer=optimizer, scheduler=scheduler, ignore_load=ignore_load)
        elif load_step:
            global_step = saverloader.load(init_dir, model.module, ignore_load=ignore_load)
        else:
            _ = saverloader.load(init_dir, model.module, ignore_load=ignore_load)
            global_step = 0
    
    requires_grad(parameters, True)
    model.train()

    pools_t = create_pools(n_pool)
    if val_freq:
        pools_v = create_pools(n_pool)
    
    while global_step < max_iters:
        global_step += 1
        iter_start_time = time.time()
        iter_rtime = 0.0
        
        for internal_step in range(grad_acc):
            read_start_time = time.time()

            if internal_step==grad_acc-1:
                sw_t = utils.improc.Summ_writer(
                    writer=writer_t,
                    global_step=global_step,
                    log_freq=log_freq,
                    fps=min(S,8),
                    scalar_freq=log_freq//4,
                    just_gif=True)
            else:
                sw_t = None

            read_new = True # read something from the dataloder
            if cache_len: 
                read_new = False
                if len(sample_pool_t) < cache_len:
                    read_new = True
                if cache_freq > 0 and global_step % cache_freq == 0:
                    read_new = True
            if read_new:
                try:
                    sample = next(iterloader_t)
                except StopIteration:
                    iterloader_t = iter(dataloader_t)
                    sample = next(iterloader_t)
                if cache_len:
                    sample_pool_t.update([sample])
                    print('cached a new sample into sample_pool (len %d)' % (len(sample_pool_t)))
            if cache_len:
                sample = sample_pool_t.sample()
                    
            iter_rtime += time.time()-read_start_time

            total_loss, metrics = run_model(
                model, sample, device,
                iters=iters,
                sw=sw_t,
                is_train=True,
                use_augs=use_augs)

            if torch.isnan(total_loss):
                print('nan in loss; quitting')
                return False

            total_loss /= grad_acc
            total_loss.backward()

        sw_t.summ_scalar('total_loss', metrics['total_loss'])

        for key in list(pools_t.keys()):
            if key in metrics:
                pools_t[key].update([metrics[key]])
            sw_t.summ_scalar('_/%s' % (key), pools_t[key].mean())

        current_lr = optimizer.param_groups[0]['lr']
        sw_t.summ_scalar('_/current_lr', current_lr)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if use_scheduler:
            scheduler.step()
        optimizer.zero_grad()

        if np.mod(global_step, save_freq)==0:
            saverloader.save(ckpt_path, optimizer, model.module, global_step, scheduler=scheduler, keep_latest=keep_latest)

        if val_freq and (global_step) % val_freq == 0:
            model.eval()
            del sample
            with torch.no_grad():
                torch.cuda.empty_cache()
            sw_v = utils.improc.Summ_writer(
                writer=writer_v,
                global_step=global_step,
                log_freq=log_freq,
                fps=min(S,8),
                scalar_freq=log_freq//4,
                just_gif=True)
            if cache_len:
                sample = sample_pool_t.sample()
            else:
                try:
                    sample = next(iterloader_t)
                except StopIteration:
                    iterloader_t = iter(dataloader_t)
                    sample = next(iterloader_t)
            with torch.no_grad():
                metrics = val_model(
                    model, sample, device,
                    iters=iters*2,
                    sw=sw_v,
                    is_train=False)
            for key in list(pools_v.keys()):
                if key in metrics:
                    pools_v[key].update([metrics[key]])
                sw_v.summ_scalar('_/%s' % (key), pools_v[key].mean())
            model.train()
                
                    
        iter_itime = time.time()-iter_start_time

        if val_freq:
            print('%s; step %06d/%d; rtime %.2f; itime %.2f; loss %.3f; loss_t %.2f; d_t %.1f; d_v %.1f' % (
                model_name, global_step, max_iters, iter_rtime, iter_itime,
                total_loss.item(), pools_t['total_loss'].mean(), pools_t['d_avg'].mean(), pools_v['d_avg'].mean()))
        else:
            print('%s; step %06d/%d; rtime %.2f; itime %.2f; loss %.3f; loss_t %.2f; d_t %.1f' % (
                model_name, global_step, max_iters, iter_rtime, iter_itime,
                total_loss.item(), pools_t['total_loss'].mean(), pools_t['d_avg'].mean()))
            
    writer_t.close()
    if val_freq:
        writer_v.close()

if __name__ == '__main__':
    Fire(main)
