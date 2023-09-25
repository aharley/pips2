import time
import numpy as np
import saverloader
from nets.pips2 import Pips
import utils.improc
import utils.misc
import random
from utils.basic import print_, print_stats
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fire import Fire
from torch.utils.data import Dataset, DataLoader
from datasets.tapviddataset_fullseq import TapVidDavis

def create_pools(n_pool=1000):
    pools = {}
    pool_names = [
        'd_1',
        'd_2',
        'd_4',
        'd_8',
        'd_16',
        'd_avg',
        'median_l2',
        'survival',
        'ate_all',
        'ate_vis',
        'ate_occ',
        'total_loss',
    ]
    for pool_name in pool_names:
        pools[pool_name] = utils.misc.SimplePool(n_pool, version='np')
    return pools

def test_on_fullseq(model, d, sw, iters=8, S_max=8, image_size=(384,512)):
    metrics = {}

    rgbs = d['rgbs'].cuda().float() # B,S,C,H,W
    trajs_g = d['trajs'].cuda().float() # B,S,N,2
    valids = d['valids'].cuda().float() # B,S,N

    B, S, C, H, W = rgbs.shape
    B, S, N, D = trajs_g.shape
    assert(D==2)
    assert(B==1)
    # print('this video is %d frames long' % S)

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
        trajs_e[:, end_frame:] = trajs_e[:, end_frame-1:end_frame] # update the future with new zero-vel

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
    median_l2 = utils.basic.reduce_masked_median(dists_, valids_, keep_batch=True)
    metrics['median_l2'] = median_l2.mean().item()

    if sw is not None and sw.save_this:
        prep_rgbs = utils.improc.preprocess_color(rgbs)
        rgb0 = sw.summ_traj2ds_on_rgb('', trajs_g[0:1], prep_rgbs[0:1,0], valids=valids[0:1], cmap='winter', linewidth=2, only_return=True)
        sw.summ_traj2ds_on_rgb('2_outputs/trajs_e_on_rgb0', trajs_e[0:1], utils.improc.preprocess_color(rgb0), valids=valids[0:1], cmap='spring', linewidth=2, frame_id=d_avg)
        st = 4
        sw.summ_traj2ds_on_rgbs2('2_outputs/trajs_e_on_rgbs2', trajs_e[0:1,::st], valids[0:1,::st], prep_rgbs[0:1,::st], valids=valids[0:1,::st], frame_ids=list(range(0,S,st)))

    return metrics

def main(
        B=1, # batchsize 
        S=128, # seqlen
        stride=8, # spatial stride of the model 
        iters=16, # inference steps of the model
        image_size=(512,896), # input resolution
        shuffle=False, # dataset shuffling
        log_freq=99, # how often to make image summaries
        max_iters=30, # how many samples to test
        log_dir='./logs_test_on_tap',
        dataset_location='/orion/u/aharley/datasets/tapvid_davis',
        init_dir='./reference_model',
        device_ids=[0], 
        n_pool=1000, # how long the running averages should be
):
    device = 'cuda:%d' % device_ids[0]

    # the idea in this file is:
    # load a ckpt, and test it in tapvid,
    # tracking points from frame0 to the end.
    
    exp_name = 'tap00' # copy from dev repo
    exp_name = 'tap01' # clean up

    assert(B==1) # B>1 not implemented here
    assert(image_size[0] % 32 == 0)
    assert(image_size[1] % 32 == 0)
    
    # autogen a descriptive name
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

    utils.misc.count_parameters(model)

    _ = saverloader.load(init_dir, model.module)
    model.eval()

    pools_x = create_pools(n_pool)

    global_step = 0
    max_iters = min(max_iters, len(dataset_x))
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
            scalar_freq=1,
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
        
        print('%s; step %06d/%d; rtime %.2f; itime %.2f; d_x %.1f; sur_x %.1f; med_x %.1f' % (
            model_name, global_step, max_iters, iter_rtime, iter_itime,
            pools_x['d_avg'].mean(), pools_x['survival'].mean(), pools_x['median_l2'].mean()))
            
    writer_x.close()
            
if __name__ == '__main__':
    Fire(main)
