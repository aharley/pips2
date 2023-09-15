import time
import numpy as np
import timeit
import utils.improc
import utils.geom
import random
from utils.basic import print_, print_stats
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from datasets.pointodysseydataset import PointOdysseyDataset
import torch.nn.functional as F
from fire import Fire
import sys
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# random.seed(125)
# np.random.seed(125)
# torch.manual_seed(125)

def run_model(d, exp_out_dir, device, mname, export=True, sw=None):
    metrics = {}

    rgbs = d['rgbs'].to(device).float().unsqueeze(0) # B,S,C,H,W
    trajs = d['trajs'].to(device).float().unsqueeze(0) # B,S,N,2
    visibs = d['visibs'].to(device).float().unsqueeze(0) # B,S,N
    valids = d['valids'].to(device).float().unsqueeze(0) # B,S,N

    B,S,C,H,W = rgbs.shape
    assert(C==3)
    
    B,S,N,D = trajs.shape
    assert(D==2)
    
    if torch.sum(valids)<B*S*N//2:
        sys.stdout.write('x')
        sys.stdout.flush()
        return False

    track_g = torch.cat([trajs, visibs.unsqueeze(-1), valids.unsqueeze(-1)], dim=3)

    temp_dir = 'temp_%s' % mname
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    npz_out_f = '%s/track.npz' % (exp_out_dir)
    rgb_out_f = '%s/rgb.mp4' % (exp_out_dir)

    if (export and
        os.path.isfile(npz_out_f) and
        os.path.isfile(rgb_out_f)):
        sys.stdout.write(':')
        sys.stdout.flush()
        return False
    
    if sw is not None and sw.save_this:
        prep_rgbs = utils.improc.preprocess_color(rgbs)
        prep_grays = prep_rgbs.mean(dim=2, keepdim=True).repeat(1,1,3,1,1)
        sw.summ_traj2ds_on_rgb('0_inputs/trajs_on_rgbs', trajs[0:1], utils.improc.preprocess_color(rgbs[0:1].mean(dim=1)), cmap='winter', linewidth=2)
        sw.summ_traj2ds_on_rgbs2('0_inputs/trajs_on_rgbs2', trajs[0:1,::4], visibs[0:1,::4], prep_rgbs[0:1,::4], valids=valids[0:1,::4], frame_ids=list(range(0,S,4)))
        # sw.summ_traj2ds_on_rgbs('0_inputs/trajs_on_rgbs', trajs[0:1], utils.improc.preprocess_color(rgbs[0:1].mean(dim=1)), cmap='winter', linewidth=2)

    rgbs = rgbs[0].byte().cpu().numpy() # S,3,H,W
    track_g = track_g[0].cpu().numpy() # S,N,2

    rgbs = rgbs.transpose(0,2,3,1) # channels last

    if export:
        if not os.path.exists(exp_out_dir):
            os.makedirs(exp_out_dir)
        
        for si in range(S):
            temp_out_f = '%s/%03d.jpeg' % (temp_dir, si)
            im = Image.fromarray(rgbs[si])
            im.save(temp_out_f, "JPEG")
        os.system('/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate 24 -pattern_type glob -i "./%s/*.jpeg" -c:v libx264 -crf 20 -pix_fmt yuv420p %s' % (temp_dir, rgb_out_f))

        # save npz only if we made all the way through
        if os.path.isfile(rgb_out_f):
            np.savez_compressed(
                npz_out_f,
                track_g=track_g,
            )
            sys.stdout.write('.')
            sys.stdout.flush()
            return True
        else:
            sys.stdout.write('f')
            sys.stdout.flush()
            return False
    return False

def main(
        dset='train', 
        S=36, # seqlen
        N=128, # number of particles to export per clip
        crop_size=(384,512), 
        use_augs=True, # resizing/jittering/color/blur augs
        shuffle=False, # dataset shuffling
        log_dir='./logs_export_mp4_dataset',
        dataset_location='/orion/group/point_odyssey',
        max_iters=0,
        log_freq=100,
        device_ids=[0],
):
    device = 'cpu:%d' % device_ids[0]

    # the idea in this file is:
    # walk through the dataset,
    # and export rgb mp4s for all valid samples,
    # so that dataloading is not a bottleneck in training
    
    exp_name = 'em00' # copy from dev repo
    
    mod = 'aa' # copy from dev repo; crop_size=(256x384), S=36
    mod = 'ab' # allow more OOB, by updating threshs to 64; export at 384,512; output 256; export as long as we have N//2
    mod = 'ac' # N=128
    mod = 'ad' # put more info into name; also print rtime
    mod = 'ae' # allow trajs to go behind camera during S

    assert(crop_size[0] % 64 == 0)
    assert(crop_size[1] % 64 == 0)
    
    # autogen a descriptive name
    model_name = "%d" % (S)
    if use_augs:
        model_name += "_A"
    model_name += "_%s" % exp_name
    model_name += "_%s" % mod
    import datetime
    model_date = datetime.datetime.now().strftime('%H%M%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    dataset_t = PointOdysseyDataset(
        dataset_location=dataset_location,
        dset=dset,
        S=S,
        N=N,
        use_augs=use_augs,
        crop_size=crop_size,
        verbose=True,
    )
    dataloader_t = DataLoader(
        dataset_t,
        batch_size=1,
        shuffle=shuffle,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        drop_last=True)
    iterloader_t = iter(dataloader_t)

    if max_iters==0:
        max_iters = len(dataloader_t) # number of samples to export

    perm = np.random.permutation(max_iters) # write in random order, for parallel
    # perm = np.arange(max_iters) # write in sequential order, for debug

    global_step = 0
    while global_step < max_iters:
        global_step += 1
        this_step = perm[global_step-1]
        iter_start_time = time.time()
        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=this_step,
            log_freq=log_freq,
            fps=4,
            scalar_freq=int(log_freq/4),
            just_gif=True)

        H, W = crop_size
        out_dir = './pod_export/%s_%d_%d_%dx%d' % (mod, S, N, H, W)
        exp_out_dir = '%s/%06d' % (out_dir, this_step)

        npz_out_f = '%s/track.npz' % (exp_out_dir)
        rgb_out_f = '%s/rgb.mp4' % (exp_out_dir)

        if (os.path.isfile(npz_out_f) and
            os.path.isfile(rgb_out_f)):
            sys.stdout.write(':')
            sys.stdout.flush()
        else:
            sample, gotit = dataset_t.__getitem__(this_step % len(dataloader_t))
            if gotit:
                iter_rtime = time.time()-iter_start_time
                out = run_model(sample, exp_out_dir, device, model_name, sw=sw_t)
                iter_itime = time.time()-iter_start_time
                if out:
                    print('%s; step %06d/%d; this_step %06d; rtime %.2f; itime %.2f' % (
                        model_name, global_step, max_iters, this_step, iter_rtime, iter_itime))
    writer_t.close()
            

if __name__ == '__main__':
    Fire(main)
