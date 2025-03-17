import os
import numpy as np
import torch
import lietorch
import droid_backends
import matplotlib.pyplot as plt

from torch.multiprocessing import Value
from lietorch import SE3
from droid_net import cvx_upsample
import geom.projective_ops as pops
from dg_slam.pose_transform import quaternion_to_transform_noBatch
from dg_slam.warp.depth_warp import depth_warp_to_mask

class DepthVideo:
    def __init__(self, image_size=[480, 640], buffer=1024, stereo=False, device="cuda:0"):
                
        # current keyframe count
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]

        self.ht_8 = self.ht // 8
        self.wd_8 = self.wd // 8

        self.coords0 = pops.coords_grid(self.ht_8, self.wd_8, device=device)

        ### state attributes ###
        self.tstamp = torch.zeros(buffer, device="cuda", dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device="cuda", dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.red = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.float).share_memory_()
        self.depths_gt = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device="cuda", dtype=torch.float).share_memory_()
        self.poses_gt = torch.zeros(buffer, 7, device="cuda", dtype=torch.float32).share_memory_()

        self.seg_masks = torch.zeros(buffer, ht//8, wd//8, device="cuda", dtype=torch.bool).share_memory_()
        self.seg_masks_ori = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.bool).share_memory_()

        self.use_segmask = True
        self.use_depth_warp = True
        self.use_depth_mask = True

        self.stereo = stereo
        c = 1 if not self.stereo else 2

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")

        self.depth_warp_mask = {}
        self.depth_warp_mask_ori = {}
        self.get_depth_warp_mask_ori = False
        
    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]
        
        if item[3] is not None:
            if len(item[3].shape) > 2:
                depth = item[3][:,3::8,3::8]
            else:
                depth = item[3][3::8,3::8]
            self.disps[index] = torch.where(depth>0, 1.0/depth, depth)

        if item[4] is not None:
            if len(item[4].shape) > 2:
                depth = item[4][:,3::8,3::8]
            else:
                depth = item[4][3::8,3::8]
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, depth)

            self.depths_gt[index] = item[4]

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            if item[7] is not None:
                self.nets[index] = item[7]

        if len(item) > 8:
            if item[8] is not None:
                self.inps[index] = item[8]
        
        if len(item) > 9:
            if item[9] is not None:
                if len(item[9].shape) > 2:
                    seg_mask = item[9][:, 3::8,3::8]
                else:
                    seg_mask = item[9][3::8,3::8]

                self.seg_masks[index] = seg_mask
                self.seg_masks_ori[index] = item[9]

        if len(item) > 10:
            self.poses_gt[index] = item[10]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False, use_mask = False):
        """ dense bundle adjustment (DBA) """

        with self.get_lock():
            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            if self.use_segmask:
                seg_masks = self.seg_masks[ii]
                seg_masks = ~ seg_masks

            if self.use_depth_warp and use_mask:
                warp_mask = []
                for id in range(len(ii)):
                    i = ii[id]
                    j = jj[id]
                    key = str(i.item()) + "to" + str(j.item())

                    cur_depth = self.depths_gt[j][3::8, 3::8].unsqueeze(0)
                    last_depth = self.depths_gt[i][3::8, 3::8].unsqueeze(0)

                    cur_pose = quaternion_to_transform_noBatch(SE3(self.poses[j]).inv().data).unsqueeze(0).clone()
                    last_pose = quaternion_to_transform_noBatch(SE3(self.poses[i]).inv().data).unsqueeze(0).clone()
                    mask_i = depth_warp_to_mask(cur_pose, last_pose ,cur_depth.unsqueeze(-1), last_depth.unsqueeze(-1) ,cur_depth ,self.intrinsics[0],self.ht_8, self.wd_8)
                    self.depth_warp_mask[key] = mask_i

                    if self.get_depth_warp_mask_ori:
                        cur_depth_ori = self.depths_gt[j].unsqueeze(0)
                        last_depth_ori = self.depths_gt[i].unsqueeze(0)
                        mask_i_ori = depth_warp_to_mask(cur_pose, last_pose, cur_depth_ori.unsqueeze(-1), last_depth_ori.unsqueeze(-1), cur_depth_ori , self.intrinsics[0] * 8, self.ht, self.wd, threshold=0.7)
                        self.depth_warp_mask_ori[key] = mask_i_ori

                    warp_mask.append(mask_i)                
                warp_mask = torch.cat(warp_mask, dim=0)

            if use_mask:
                final_mask = (seg_masks | warp_mask)
                weight = weight * final_mask.unsqueeze(1)
            else:
                final_mask = seg_masks
                weight = weight * final_mask.unsqueeze(1)

            droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)

            self.disps.clamp_(min=0.001)