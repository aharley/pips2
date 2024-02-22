import time
import numpy as np
import timeit
import saverloader
import utils.improc
import utils.geom
import utils.misc
import utils.vox
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

# define the scene centroid for 3d vox
scene_centroid_x = 0.0 # right
scene_centroid_y = 0.0 # down
scene_centroid_z = 4.0 # forward
scene_centroid_py = np.array([scene_centroid_x,
                              scene_centroid_y,
                              scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid_py).float()

# define a volume around the centroid
XMIN, XMAX = -4, 4
ZMIN, ZMAX = -4, 4
YMIN, YMAX = -4, 4
bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

Z, Y, X = 500, 50, 500

def run_model(d, device, sw=None):
    rgbs = d['rgbs'].to(device).float() # B,S,3,H,W
    depths = d['depths'].to(device).float() # B,S,1,H,W
    normals = d['normals'].to(device).float() # B,S,3,H,W
    trajs_world = d['trajs_world'].to(device).float() # B,S,N,3
    trajs_x = d['trajs_pix'].to(device).float() # B,S,N,2
    trajs_g = d['trajs_2d'].to(device).float() # B,S,N,2
    vis_g = d['visibs'].to(device).float() # B,S,N
    valids = d['valids'].to(device).float() # B,S,N

    pix_T_cams = d['pix_T_cams'].to(device).float() # B,S,4,4
    cams_T_world = d['cams_T_world'].to(device).float() # B,S,4,4

    B, S, C, H, W = rgbs.shape
    assert(C==3)
    B, S, N, D = trajs_g.shape
    assert(D==2)

    assert(B==1)

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)
    
    print_stats('depths', depths)
    depths_valid = (depths > 0.0).float()
    med_depth = utils.basic.reduce_masked_median(depths, depths_valid)
        
    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)
    
    trajs_cam = __u(utils.geom.apply_4x4(__p(cams_T_world), __p(trajs_world)))
    trajs_pix = __u(utils.geom.apply_pix_T_cam(__p(pix_T_cams), __p(trajs_cam)))

    cam0_T_world = cams_T_world[:,0]

    occ0_vis = []
    occI_vis = []
    for si in range(S):
        world_T_camI = utils.geom.safe_inverse(cams_T_world[:,si])
        cam0_T_camI = utils.geom.matmul2(cam0_T_world, world_T_camI)

        xyz_camI = utils.geom.depth2pointcloud(depths[:,si], pix_T_cams[:,si])
        depth_valid_ = depths_valid[:,si].reshape(-1)
        xyz_camI = xyz_camI[:,depth_valid_>0]
        xyz_cam0 = utils.geom.apply_4x4(cam0_T_camI, xyz_camI)

        occ_memI = vox_util.voxelize_xyz(xyz_camI, Z, Y, X, assert_cube=False)
        occI_vis.append(sw.summ_occ('', occ_memI, only_return=True))

        occ_mem0 = vox_util.voxelize_xyz(xyz_cam0, Z, Y, X, assert_cube=False)
        occ0_vis.append(sw.summ_occ('', occ_mem0, only_return=True))

        if sw is not None and sw.save_this:
            sw.summ_rgbs('0_inputs/occI_vis', occI_vis)
            sw.summ_rgbs('0_inputs/occ0_vis', occ0_vis)
            sw.summ_rgb('0_inputs/normal0', utils.basic.normalize(normals[:,0])-0.5)
            prep_rgbs = utils.improc.preprocess_color(rgbs)

            sw.summ_traj2ds_on_rgb('0_inputs/trajs_x_on_rgb', trajs_x[0:1], prep_rgbs.mean(dim=1), valids=valids[0:1], cmap='winter')
            sw.summ_traj2ds_on_rgb('0_inputs/trajs_g_on_rgb', trajs_g[0:1], prep_rgbs.mean(dim=1), valids=valids[0:1], cmap='winter')
            sw.summ_traj2ds_on_rgb('0_inputs/trajs_pix_on_rgb', trajs_pix[0:1], prep_rgbs.mean(dim=1), valids=valids[0:1], cmap='winter')

            sw.summ_rgb('0_inputs/rgb0', rgbs[:,0].byte(), frame_id=med_depth.item())
            sw.summ_oned('0_inputs/depth0', depths[:,0] * depths_valid[:,0], max_val=scene_centroid_z+ZMAX, frame_id=med_depth.item())
        
        
    return None 
    

def main(
        exp_name='debug',
        dset='train',
        B=1, # batchsize 
        S=5, # seqlen
        N=1024, # number of points per clip
        use_augs=False, # resizing/jittering/color/blur augs
        shuffle=False, # dataset shuffling
        log_dir='./logs_just_vis_3d',
        dataset_location='/orion/group/point_odyssey_v1.2',
        log_freq=1,
        max_iters=10,
        quick=False,
        verbose=True,
        dname=None,
):
    device = 'cpu:0'

    # the idea in this file is:
    # load the 3d pointodyssey data and visualize it
    
    exp_name = 'jw00' # copy from dev repo
    exp_name = 'jw01' # rescale depths and extrinsics
    exp_name = 'jw02' # clean up for v1.2

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
        verbose=verbose,
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

        iter_rtime = time.time()-iter_start_time
        _ = run_model(sample, device, sw=sw_t)
        iter_itime = time.time()-iter_start_time
        
        print('%s; step %06d/%d; rtime %.2f; itime %.2f' % (
            model_name, global_step, max_iters, iter_rtime, iter_itime))
            
    writer_t.close()
            

if __name__ == '__main__':
    Fire(main)
