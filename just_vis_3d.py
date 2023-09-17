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
    trajs_x = d['trajs_pix'].to(device).float() # B,S,N,2
    trajs_g = d['trajs_2d'].to(device).float() # B,S,N,2
    vis_g = d['visibs'].to(device).float() # B,S,N
    valids = d['valids'].to(device).float() # B,S,N

    pix_T_cams = d['pix_T_cams'].to(device).float() # B,S,4,4
    cams_T_world = d['cams_T_world'].to(device).float() # B,S,4,4

    print('rgbs', rgbs.shape)
    print('trajs_x', trajs_x.shape)
    print('trajs_g', trajs_g.shape)
    print('vis_g', vis_g.shape)
    print('valids', valids.shape, torch.sum(valids[:,0]))
    print('pix_T_cams', pix_T_cams.shape)
    print('cams_T_world', cams_T_world.shape)

    B, S, C, H, W = rgbs.shape
    assert(C==3)
    B, S, N, D = trajs_g.shape
    assert(D==2)

    assert(B==1)


    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)
    
    # if torch.median(depth) > 100:

    normals = utils.improc.preprocess_color(normals)
    
    print_stats('normals', normals)

    depths_valid = (depths < 1000.0).float() * (depths > 0.0).float()

    # med = utils.basic.reduce_masked_median(depths[:,0], depths_valid[:,0])
    med = utils.basic.reduce_masked_median(depths, depths_valid)
    # med = torch.median(depths[:,0])
    print_('masked median depth', med)

    if med > 10:
        xyz_cam0 = utils.geom.depth2pointcloud(depths[:,0], pix_T_cams[:,0])
        # print('xyz_cam0[0,:10] bef', xyz_cam0[0,:10])
        
        depths = depths / 12.0
        xyz_cam0 = utils.geom.depth2pointcloud(depths[:,0], pix_T_cams[:,0])
        # print('xyz_cam0[0,:10] aft', xyz_cam0[0,:10])
        # input()

        # so,
        # it seems all of the values are divided by 12
        # which makes perfect sense via the math


        # what i need to do is:
        # undo the scale temporarily, apply the extrinsics, and redo it
        
        # sc = utils.geom.eye_3x3(B*S, device=device)
        sc = utils.geom.eye_4x4(B*S, device=device)
        sc[:,:3,:3] /= 12.0
        
        # print_('sc[0]', sc[0])
        # print_('inverse(sc[0])', sc.inverse()[0])
        # print_stats('cams_T_world before', cams_T_world)
        
        # cams_T_world = __u(utils.geom.matmul2(sc, __p(cams_T_world)))
        cams_T_world = __u(utils.geom.matmul3(sc, __p(cams_T_world), sc.inverse()))

        # print_stats('cams_T_world after', cams_T_world)
        
        # print_('sc[0]', sc[0])
        # # cams_T_world = __u(utils.geom.matmul2(sc, __p(cams_T_world)))
        # pix_T_cams = __u(utils.geom.matmul2(sc, __p(pix_T_cams)))
        # pix_T_cams = __u(utils.geom.matmul2(__p(pix_T_cams), sc))

        
        # cams_T_world = __u(utils.geom.matmul2(sc, __p(cams_T_world)))

        # med = utils.basic.reduce_masked_median(depths, depths_valid)
        # # med = torch.median(depths[:,0])
        # print_('new masked median depth', med)

    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)

    # world_T_cams = __u(utils.geom.safe_inverse(__p(cams_T_world)))
    # utils.geom.get_camM_T_camXs(origin_T_camXs, ind=0)
    # cam0_T_camXs = utils.geom.get_camM_T_camXs(velo_T_cams, ind=0)
    # camXs_T_cam0 = __u(utils.geom.safe_inverse(__p(cam0_T_camXs)))

    cam0_T_world = cams_T_world[:,0]

    occ0_vis = []
    occI_vis = []
    for si in range(S):

        # cam0_T_camXs = utils.geom.get_camM_T_camXs(velo_T_cams, ind=0)

        # camI_T_world = cams_T_world[:,si]
        world_T_camI = utils.geom.safe_inverse(cams_T_world[:,si])
        cam0_T_camI = utils.geom.matmul2(cam0_T_world, world_T_camI)

        xyz_camI = utils.geom.depth2pointcloud(depths[:,si], pix_T_cams[:,si])
        depth_valid_ = depths_valid[:,si].reshape(-1)
        xyz_camI = xyz_camI[:,depth_valid_>0]
        xyz_cam0 = utils.geom.apply_4x4(cam0_T_camI, xyz_camI)

        if si==0:
            print_stats('xyz_camI', xyz_camI)
            print_stats('xyz_cam0', xyz_cam0)

        occ_memI = vox_util.voxelize_xyz(xyz_camI, Z, Y, X, assert_cube=False)
        occI_vis.append(sw.summ_occ('', occ_memI, only_return=True))

        occ_mem0 = vox_util.voxelize_xyz(xyz_cam0, Z, Y, X, assert_cube=False)
        occ0_vis.append(sw.summ_occ('', occ_mem0, only_return=True))


        # print_stats('depths', depths)
        # max_depth = 16
        # depths_valid = (depths < max_depth).float() * (depths > 0.01).float()
        # depths = depths * depths_valid
        # print_stats('depths', depths)

        if sw is not None and sw.save_this:
            sw.summ_rgbs('0_inputs/occI_vis', occI_vis)
            sw.summ_rgbs('0_inputs/occ0_vis', occ0_vis)
            # sw.summ_rgb('0_inputs/normal0', utils.basic.normalize(normals[:,0])-0.5)
            # sw.summ_rgb('0_inputs/normal0', normals[:,0])
            sw.summ_rgbs('0_inputs/normals', normals[:,:4].unbind(1))
            # prep_rgbs = utils.improc.preprocess_color(rgbs)
            # prep_grays = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)

            # sw.summ_traj2ds_on_rgb('0_inputs/trajs_x_on_rgb', trajs_x[0:1], prep_rgbs.mean(dim=1), valids=valids[0:1], cmap='winter')
            # sw.summ_traj2ds_on_rgb('0_inputs/trajs_g_on_rgb', trajs_g[0:1], prep_rgbs.mean(dim=1), valids=valids[0:1], cmap='winter')

            # sw.summ_oned('0_inputs/depth0', depths[:,0], norm=True, frame_id=med.item())
            sw.summ_rgb('0_inputs/rgb0', rgbs[:,0].byte(), frame_id=med.item())
            # sw.summ_oned('0_inputs/depth0', depths[:,0] * depths_valid[:,0], norm=True, frame_id=med.item())
            # sw.summ_oned('0_inputs/depth0', depths[:,0] * depths_valid[:,0], norm=False, max_val=8.0, frame_id=med.item())

            print_stats('depths[:,0]', depths[:,0])

            sw.summ_oned('0_inputs/depth0', depths[:,0] * depths_valid[:,0], max_val=scene_centroid_z+ZMAX, frame_id=med.item())
        
        
    return None 
    

def main(
        exp_name='debug',
        dset='train',
        B=1, # batchsize 
        S=8, # seqlen
        N=256, # number of points per clip
        use_augs=False, # resizing/jittering/color/blur augs
        shuffle=False, # dataset shuffling
        log_dir='./logs_just_vis_3d',
        dataset_location='/orion/group/point_odyssey',
        log_freq=1,
        max_iters=10,
        quick=False,
        dname=None,
):
    device = 'cpu:0'

    # the idea in this file is:
    # load the 3d pointodyssey data and visualize it
    
    exp_name = 'jw00' # copy from dev repo
    exp_name = 'jw01' # clean up
    exp_name = 'jw02' # go
    exp_name = 'jw03' # collect vis*valid first
    exp_name = 'jw04' # show me depth
    exp_name = 'jw05' # print median
    exp_name = 'jw06' # scale extrinsics

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
