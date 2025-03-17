#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from dg_slam.gaussian.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from dg_slam.gaussian.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from dg_slam.gaussian.sh_utils import RGB2SH, SH2RGB, eval_sh
from simple_knn._C import distCUDA2
from dg_slam.gaussian.graphics_utils import BasicPointCloud
from dg_slam.gaussian.general_utils import strip_symmetric, build_scaling_rotation

import faiss
import faiss.contrib.torch_utils
from dg_slam.gaussian.common import setup_seed, clone_kf_dict

class GaussianModel():
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, cfg):
        sh_degree = cfg["gaussian"]["sh_degree"]
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self._pts_num = 0        # number of points in neural point cloud
        self.geo_feats = None
        self.col_feats = None
        self.max_sh_degree = 3

        self.cfg = cfg
        self.c_dim = cfg['model']['c_dim']
        self.device = cfg['mapping']['device']
        self.cuda_id = 0
        self.use_dynamic_radius = cfg['use_dynamic_radius']
        self.nn_num = cfg['pointcloud']['nn_num']

        self.nlist = cfg['pointcloud']['nlist']
        self.radius_add = cfg['pointcloud']['radius_add']
        self.radius_min = cfg['pointcloud']['radius_min']
        self.radius_query = cfg['pointcloud']['radius_query']
        self.fix_interval_when_add_along_ray = cfg['pointcloud']['fix_interval_when_add_along_ray']

        self.N_surface = cfg['rendering']['N_surface']
        self.N_add = cfg['pointcloud']['N_add']
        self.near_end_surface = cfg['pointcloud']['near_end_surface']
        self.far_end_surface = cfg['pointcloud']['far_end_surface']

        self._cloud_pos = [] 
        self._pts_num = 0        # number of points in neural point cloud
        self.geo_feats = None
        self.col_feats = None
        self.keyframe_dict = []

        self.geo_feats = None
        self.col_feats = None

        self.resource = faiss.StandardGpuResources() 
        self.index = faiss.index_cpu_to_gpu(self.resource,
                                            self.cuda_id,
                                            faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, self.nlist, faiss.METRIC_L2))
        self.index.nprobe = cfg['pointcloud']['nprobe']
        setup_seed(cfg["setup_seed"])
    
    def get_scaling(self):
        return self._scaling
    
    def get_rotation(self):
        return self._rotation
    
    def get_xyz(self):
        return self._xyz
    
    def get_features_dc(self):
        return self._features_dc
    
    def get_features_rest(self):
        return self._features_rest

    def get_gaussian_opt(self):
        return self.optimizer

    def get_opacity(self):
        return self._opacity
    
    def get_scaling(self):
        return self._scaling
    
    def get_active_sh_degree(self):
        return self.active_sh_degree
    
    def get_max_sh_degree(self):
        return self.max_sh_degree
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def add_neural_points(self, batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color,\
                           train=False, is_pts_grad=False, dynamic_radius=None):
        """
        Add multiple neural points, will use depth filter when getting these samples.

        Args:
            batch_rays_o (tensor): ray origins (N,3)
            batch_rays_d (tensor): ray directions (N,3)
            batch_gt_depth (tensor): sensor depth (N,)
            batch_gt_color (tensor): sensor color (N,3)
            train (bool): whether to update the FAISS index
            is_pts_grad (bool): the points are chosen based on color gradient
            dynamic_radius (tensor): choose every radius differently based on its color gradient

        """

        if batch_rays_o.shape[0]:
            mask = batch_gt_depth > 0
            batch_gt_color = RGB2SH(batch_gt_color)
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = \
                batch_rays_o[mask], batch_rays_d[mask], batch_gt_depth[mask], batch_gt_color[mask]

            pts_gt = batch_rays_o[..., None, :] + batch_rays_d[..., None, :] * batch_gt_depth[..., None, None]
            mask = torch.ones(pts_gt.shape[0], device=self.device).bool()
            pts_gt = pts_gt.reshape(-1, 3)

            if self.index.is_trained: 
                _, _, neighbor_num_gt = self.find_neighbors_faiss(
                    pts_gt, step='add', is_pts_grad=is_pts_grad, dynamic_radius=dynamic_radius)
                mask = (neighbor_num_gt == 0)

            gt_depth_surface = batch_gt_depth.unsqueeze(
                -1).repeat(1, self.N_add)
            t_vals_surface = torch.linspace(
                0.0, 1.0, steps=self.N_add, device=self.device)

            if self.N_add == 1:
                z_vals = gt_depth_surface

            pts = batch_rays_o[..., None, :] + \
                batch_rays_d[..., None, :] * z_vals[..., :, None]
            pts = pts[mask]  # use mask from pts_gt for auxiliary points
            pts = pts.reshape(-1, 3)

            self._cloud_pos += pts.tolist()
            self._pts_num += pts.shape[0]

            batch_gt_color = batch_gt_color.unsqueeze(1).repeat(1,self.N_add,1)
            batch_gt_color = batch_gt_color[mask].reshape(-1, 3)
            features = torch.zeros((batch_gt_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            
            features[:, :3, 0 ] = batch_gt_color
            features[:, 3:, 1:] = 0.0

            if train or not self.index.is_trained:
                self.index.train(pts)
            self.index.train(torch.tensor(self._cloud_pos, device=self.device))
            self.index.add(pts)

            dist4, _ = self.find_closed_faiss(pts)
            dist4 = dist4[:, 1:].mean(dim=1)
            dist4 = torch.clamp_min(dist4.float().cuda(), 0.0000001)
            scales = torch.log(torch.sqrt(dist4))[..., None].repeat(1, 3)

            rots = torch.zeros((pts.shape[0], 4), device="cuda")
            rots[:, 0] = 1

            opacities = inverse_sigmoid(0.1 * torch.ones((pts.shape[0], 1), dtype=torch.float, device="cuda"))

            if self._xyz is None:
                self._xyz = pts
                self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
                self._features_rest = features[:,:,1:].transpose(1, 2).contiguous()
                self._scaling = scales
                self._rotation = rots
                self._opacity = opacities
                self.max_radii2D = torch.zeros((pts.shape[0]), device="cuda")
            else:
                self._xyz = torch.cat((self._xyz, pts), dim=0)
                self._features_dc = torch.cat((self._features_dc, features[:,:,0:1].transpose(1, 2).contiguous()), dim=0)
                self._features_rest = torch.cat((self._features_rest, features[:,:,1:].transpose(1, 2).contiguous()), dim=0)
                self._scaling = torch.cat((self._scaling, scales), dim=0)
                self._rotation = torch.cat((self._rotation, rots), dim=0)
                self._opacity = torch.cat((self._opacity, opacities), dim=0)
                self.max_radii2D = torch.cat((self.max_radii2D, torch.zeros((pts.shape[0]), device="cuda")), dim=0)

            return torch.sum(mask)
        else:
            return 0

    def update_xyz(self, feats, indices=None):
        assert torch.is_tensor(feats), 'use tensor to update features'
        if indices is not None:
            self._xyz[indices] = feats.detach().clone()
        else:
            self._xyz = feats.detach().clone()

        self._cloud_pos = self._xyz.tolist()

    def update_features_dc(self, feats, indices=None):
        assert torch.is_tensor(feats), 'use tensor to update features'
        if indices is not None:
            self._features_dc[indices] = feats.detach().clone()
        else:
            self._features_dc = feats.detach().clone()

    def update_features_rest(self, feats, indices=None):
        assert torch.is_tensor(feats), 'use tensor to update features'
        if indices is not None:
            self._features_rest[indices] = feats.detach().clone()
        else:
            self._features_rest = feats.detach().clone()

    def update_scaling(self, feats, indices=None):
        assert torch.is_tensor(feats), 'use tensor to update features'
        if indices is not None:
            self._scaling[indices] = feats.detach().clone()
        else:
            self._scaling = feats.detach().clone()

    def update_opacity(self, feats, indices=None):
        assert torch.is_tensor(feats), 'use tensor to update features'
        if indices is not None:
            self._opacity[indices] = feats.detach().clone()
        else:
            self._opacity = feats.detach().clone()

    def update_rotation(self, feats, indices=None):
        assert torch.is_tensor(feats), 'use tensor to update features'
        if indices is not None:
            self._rotation[indices] = feats.detach().clone()
        else:
            self._rotation = feats.detach().clone()

    def update_opt(self, opt):
        self.optimizer = opt

    def input_pos(self):
        return self._xyz.detach().cpu()

    def input_rgb(self):
        sh0 = self._features_dc
        color = SH2RGB(sh0)
        return color.squeeze(1).detach().cpu()
    
    def pts_num(self):
        return self._pts_num
    
    def find_neighbors_faiss(self, pos, step='add', retrain=False, is_pts_grad=False, dynamic_radius=None):
        """
        Query neighbors using faiss.

        Args:
            pos (tensor): points to find neighbors
            step (str): 'add'|'query'
            retrain (bool, optional): if to retrain the index cluster of IVF
            is_pts_grad: whether it's the points chosen based on color grad, will use smaller radius when looking for neighbors
            dynamic_radius (tensor, optional): choose every radius differently based on its color gradient

        Returns:
            D: distances to neighbors for the positions in pos
            I: indices of neighbors for the positions in pos
            neighbor_num: number of neighbors for the positions in pos
        """
        if (not self.index.is_trained) or retrain:
            self.index.train(self._cloud_pos)

        assert step in ['add', 'query']
        split_pos = torch.split(pos, 65000, dim=0)
        D_list = []
        I_list = []
        for split_p in split_pos:
            D, I = self.index.search(split_p.float(), self.nn_num)
            D_list.append(D)
            I_list.append(I)
        D = torch.cat(D_list, dim=0)
        I = torch.cat(I_list, dim=0)

        if step == 'query':  # used if dynamic_radius is None
            radius = self.radius_query
        else:  # step == 'add', used if dynamic_radius is None
            if not is_pts_grad:
                radius = self.radius_add
            else:
                radius = self.radius_min

        # faiss returns "D" in the form of squared distances. Thus we compare D to the squared radius
        if dynamic_radius is not None:
            assert pos.shape[0] == dynamic_radius.shape[0], 'shape mis-match for input points and dynamic radius'
            neighbor_num = (D < dynamic_radius.reshape(-1, 1)
                            ** 2).sum(axis=-1).int()
        else:
            neighbor_num = (D < radius**2).sum(axis=-1).int()

        return D, I, neighbor_num

    def find_closed_faiss(self, pos, retrain=False, is_pts_grad=False):
        """
        Query neighbors using faiss.

        Args:
            pos (tensor): points to find neighbors
            retrain (bool, optional): if to retrain the index cluster of IVF
            is_pts_grad: whether it's the points chosen based on color grad, will use smaller radius when looking for neighbors

        Returns:
            D: distances to neighbors for the positions in pos
            I: indices of neighbors for the positions in pos
        """
        if (not self.index.is_trained) or retrain:
            self.index.train(self._cloud_pos)

        split_pos = torch.split(pos, 65000, dim=0)
        D_list = []
        I_list = []
        for split_p in split_pos:
            D, I = self.index.search(split_p.float(), 4)
            D_list.append(D)
            I_list.append(I)

        D = torch.cat(D_list, dim=0)
        I = torch.cat(I_list, dim=0)

        return D, I