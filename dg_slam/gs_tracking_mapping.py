import os
import cv2
import numpy as np
import open3d as o3d
import torch
from torch import nn

from colorama import Fore, Style
from torch.autograd import Variable
from dg_slam.gaussian.general_utils import inverse_sigmoid, build_rotation
from dg_slam.gaussian.common import (get_camera_from_tensor, get_samples, get_samples_with_pixel_grad, get_samples_point_add,
                        get_tensor_from_camera, setup_seed, random_select)
from dg_slam.gaussian.logger import Logger
from dg_slam.gaussian.loss_utils import ssim
from dg_slam.gaussian.gaussian_render import render
from dg_slam.gaussian.graphics_utils import getProjectionMatrix
from dg_slam.gaussian.common import focal2fov, convert3x4_4x4
from dg_slam.gaussian_model import GaussianModel
from dg_slam.warp.depth_warp import depth_warp_pixel

from skimage.color import rgb2gray
from skimage import filters
from scipy.interpolate import interp1d
import wandb

def pose_matrix_from_quaternion(pvec):
    """ convert 4x4 pose matrix to (t, q) """
    from scipy.spatial.transform import Rotation
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
    pose[:3, 3] = pvec[:3]
    return pose

class gs_tracking_mapping():
    def __init__(self, cfg, args, video):
        self.cfg = cfg
        self.args = args
        self.video = video
        self.scale_factor = 0.8

        self.low_gpu_mem = cfg['low_gpu_mem']
        self.verbose = cfg['verbose']
        self.dataset = cfg['dataset']
        self.time_string = cfg["data"]["exp_name"]
        self.output = os.path.join(cfg["data"]["output"], self.time_string)
        os.makedirs(self.output, exist_ok=True)
        self.ckptsdir = os.path.join(self.output, 'ckpts')

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.edge = None        
        self.update_scale()
        self.update_cam()

        self.gaussians = GaussianModel(cfg)

        self.n_img = cfg["data"]["n_img"]
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.idx = torch.zeros((1)).int()
        self.mapping_idx = torch.zeros((1)).int()

        self.wandb = cfg['wandb']
        self.project_name = cfg['project_name']
        self.use_dynamic_radius = cfg['use_dynamic_radius']
        self.dynamic_r_add, self.dynamic_r_query = None, None
        self.encode_exposure = cfg['model']['encode_exposure']
        self.radius_add_max = cfg['pointcloud']['radius_add_max']
        self.radius_add_min = cfg['pointcloud']['radius_add_min']
        self.radius_query_ratio = cfg['pointcloud']['radius_query_ratio']
        self.color_grad_threshold = cfg['pointcloud']['color_grad_threshold']
        self.eval_img = cfg['rendering']['eval_img']
        
        self.device = cfg['mapping']['device']
        self.fix_geo_decoder = cfg['mapping']['fix_geo_decoder']
        self.fix_color_decoder = cfg['mapping']['fix_color_decoder']
        self.eval_rec = cfg['meshing']['eval_rec']
        self.BA = cfg['mapping']['BA']
        self.BA_cam_lr = cfg['mapping']['BA_cam_lr']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.mapping_pixels = cfg['mapping']['pixels'] 
        self.pixels_adding = cfg['mapping']['pixels_adding']
        self.pixels_based_on_color_grad = cfg['mapping']['pixels_based_on_color_grad']
        self.pixels_based_on_render = cfg['mapping']['pixels_based_on_render']
        self.add_pixel_depth_th = cfg['mapping']['add_pixel_depth_th']
        self.num_joint_iters = cfg['mapping']['iters']
        self.geo_iter_first = cfg['mapping']['geo_iter_first']
        self.iters_first = cfg['mapping']['iters_first']
        self.every_frame = cfg['mapping']['every_frame']
        self.color_refine = cfg['mapping']['color_refine']
        self.w_color_loss = cfg['mapping']['w_color_loss']
        self.w_geo_loss = cfg['mapping']['w_geo_loss']
        self.lambda_ssim_loss = cfg['mapping']['lambda_ssim_loss']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.geo_iter_ratio = cfg['mapping']['geo_iter_ratio']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.frustum_feature_selection = cfg['mapping']['frustum_feature_selection']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']
        self.save_selected_keyframes_info = cfg['mapping']['save_selected_keyframes_info']
        self.frustum_edge = cfg['mapping']['frustum_edge']
        self.save_ckpts = cfg['mapping']['save_ckpts']
        self.crop_edge = 0 if cfg['cam']['crop_edge'] is None else cfg['cam']['crop_edge']
        self.min_iter_ratio = cfg['mapping']['min_iter_ratio']
        self.lazy_start = cfg['mapping']['lazy_start']

        if self.save_selected_keyframes_info:
            self.selected_keyframes = {}

        self.keyframe_dict = []
        self.keyframe_list = []

        self.logger = Logger(cfg, args, self)
        self.position_lr_init = 0.0001
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.0001
        self.rotation_lr = 0.001

        self.fovx = focal2fov(self.fx, self.W)
        self.fovy = focal2fov(self.fy, self.H)
        self.zfar = 100.0
        self.znear = 0.01
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.fovx, fovY=self.fovy).transpose(0,1).cuda()

        self.gaussians_xyz = None
        self.gaussians_features_dc = None
        self.gaussians_features_rest = None
        self.gaussians_opacity = None
        self.gaussians_scaling = None
        self.gaussians_rotation = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.gaussians_xyz_grad = None
        self.gaussians_features_dc_grad = None
        self.gaussians_features_rest_grad = None
        self.gaussians_opacity_grad = None
        self.gaussians_scaling_grad = None
        self.gaussians_rotation_grad = None

        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.max_radii2D = None
        self.optimizer = None

        self.cam_lr = cfg['tracking']['lr']
        self.num_cam_iters = cfg['tracking']['iters']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.separate_LR = cfg['tracking']['separate_LR']
        self.w_color_loss_tracking = cfg['tracking']['w_color_loss']
        self.w_geo_loss_tracking = cfg['tracking']['w_geo_loss']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.handle_dynamic = cfg['tracking']['handle_dynamic']
        self.use_color_in_tracking = cfg['tracking']['use_color_in_tracking']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']
        self.sample_with_color_grad = cfg['tracking']['sample_with_color_grad']
        self.depth_limit = cfg['tracking']['depth_limit']
        self.use_opacity_mask_for_loss = cfg['tracking']['use_opacity_mask_for_loss']
        self.ignore_outlier_depth_loss = cfg['tracking']['ignore_outlier_depth_loss']
        self.opacity_thres = cfg['tracking']['opacity_thres']

        self.prev_mapping_idx = -1
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.opacity_mask = None
        self.count_bound = 0
        self.inv_pose = None

        self.exposure_feat_all = ([] if self.encode_exposure else None)
        
        self.print_output_desc()

    def print_output_desc(self):
        print(f"⭐️ 🌙  mapping begin!!")

    def update_scale(self):
        self.H *= self.scale_factor
        self.W *= self.scale_factor
        self.H = int(self.H)
        self.W = int(self.W)
        self.fx *= self.scale_factor
        self.fy *= self.scale_factor
        self.cx *= self.scale_factor
        self.cy *= self.scale_factor

    def update_cam(self):
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']
            self.edge = self.cfg['cam']['crop_edge']

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_neural_point(self, min_opacity):
        prune_mask = (self.opacity_activation(self.gaussians_opacity_grad) < min_opacity).squeeze()
        scale_mask1 = torch.max(self.scaling_activation(self.gaussians_scaling_grad), dim=-1)[0] > 0.4
        scale_mask2 = torch.max(self.scaling_activation(self.gaussians_scaling_grad), dim=-1)[0] > \
                      (torch.min(self.scaling_activation(self.gaussians_scaling_grad), dim=-1)[0] * 36)
        prune_mask = (prune_mask | scale_mask1 | scale_mask2).detach()

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self.gaussians_xyz_grad = optimizable_tensors["xyz"]
        self.gaussians_features_dc_grad = optimizable_tensors["f_dc"]
        self.gaussians_features_rest_grad = optimizable_tensors["f_rest"]
        self.gaussians_opacity_grad = optimizable_tensors["opacity"]
        self.gaussians_scaling_grad = optimizable_tensors["scaling"]
        self.gaussians_rotation_grad = optimizable_tensors["rotation"]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups[:6]:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def get_mask_from_c2w(self, c2w, depth_np):
        """
        Frustum feature selection based on current camera pose and depth image.
        Args:
            c2w (tensor): camera pose of current frame.
            depth_np (numpy.array): depth image of current frame. for each (x,y)<->(width,height)

        Returns:
            mask (tensor): mask for selected optimizable feature.
        """
        H, W, fx, fy, cx, cy, = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        points = np.array(self.gaussians.get_xyz().cpu()).reshape(-1, 3)

        c2w = c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(points[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate(
            [points, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c@homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
        
        uv = K@cam_cord
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.astype(np.float32)

        remap_chunk = int(3e4)
        depths = []
        for i in range(0, uv.shape[0], remap_chunk):
            depths += [cv2.remap(depth_np,
                                 uv[i:i+remap_chunk, 0],
                                 uv[i:i+remap_chunk, 1],
                                 interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)]
        depths = np.concatenate(depths, axis=0)

        edge = self.frustum_edge
        mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
            (uv[:, 1] < H-edge)*(uv[:, 1] > edge)

        zero_mask = (depths == 0)
        depths[zero_mask] = np.max(depths)
        mask = mask & (0 <= z[:, :, 0]) & (z[:, :, 0] <= depths+0.5)
        mask = mask.reshape(-1)
        return np.where(mask)[0].tolist(), np.where(~mask)[0].tolist()

    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, keyframe_dict, k, N_samples=8, pixels=200):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            c2w (tensor): camera to world matrix (3*4 or 4*4 both fine).
            keyframe_dict (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            N_samples (int, optional): number of samples/points per ray. Defaults to 16.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 100.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, gt_depth, gt_color = get_samples(
            0, H, 0, W, pixels,
            fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device, depth_filter=True)

        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, N_samples)
        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        near = gt_depth*0.8
        far = gt_depth+0.5
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * z_vals[..., :, None]
        vertices = pts.reshape(-1, 3).cpu().numpy()
        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_dict):
            c2w = keyframe['est_c2w'].cpu().numpy()
            w2c = np.linalg.inv(c2w)
            ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate(
                [vertices, ones], axis=1).reshape(-1, 4, 1)
            cam_cord_homo = w2c@homo_vertices
            cam_cord = cam_cord_homo[:, :3]
            K = np.array([[fx, .0, cx], [.0, fy, cy],
                         [.0, .0, 1.0]]).reshape(3, 3)
            
            uv = K@cam_cord
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.astype(np.float32)
            edge = 20
            mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
                (uv[:, 1] < H-edge)*(uv[:, 1] > edge)
            
            mask = mask & (z[:, :, 0] > 0)  
            mask = mask.reshape(-1)
            percent_inside = mask.sum()/uv.shape[0]
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)

        selected_keyframe_list = [dic['id']
                                  for dic in list_keyframe if dic['percent_inside'] > 0.00]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k])
        return selected_keyframe_list

    def reset_opacity(self):
        get_opacity = self.opacity_activation(self.gaussians_opacity_grad)
        opacities_new = inverse_sigmoid(torch.min(get_opacity, torch.ones_like(get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self.gaussians_opacity_grad = optimizable_tensors["opacity"]        
    
    def optimize_cur_map(self, num_joint_iters, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w, cur_c2w, color_refine=False, seg_mask = None):
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        init = True if idx == 0 else False
        
        mlp_exposure_para_list = []
        gt_depth_np = cur_gt_depth.cpu().numpy()
        gt_depth = cur_gt_depth
        gt_color = cur_gt_color

        if idx == 0:
            add_pts_num = torch.clamp(self.pixels_adding * ((gt_depth.median() / 2.5) ** 2),
                                      min=self.pixels_adding, max=self.pixels_adding * 3).int().item() * 2
        else:
            add_pts_num = self.pixels_adding // 10

        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, i, j = get_samples(
            0, H, 0, W, add_pts_num,
            fx, fy, cx, cy, cur_c2w, gt_depth, gt_color, self.device, depth_filter=True, return_index=True, seg_mask = seg_mask)
        
        keyframe_len = idx
        warp_window = 3
        warp_depth_img_batch = []
        warp_pose_batch = []
        warp_gt_pose_batch = []
        if keyframe_len >= warp_window:
            for frame_id in range(1, warp_window + 1):
                query_id = frame_id * (-1)
                tmp_gt_depth = self.video.depths_gt[idx + query_id]
                tmp_gt_c2w = self.video.poses_gt[idx + query_id] 
                tmp_est_c2w = self.estimate_c2w_list[idx + query_id].cuda() 

                warp_pose_batch += [tmp_est_c2w]
                warp_depth_img_batch += [tmp_gt_depth]
                tmp_gt_c2w = torch.from_numpy(pose_matrix_from_quaternion(tmp_gt_c2w.cpu())).cuda()
                warp_gt_pose_batch += [tmp_gt_c2w]
        else:
            for frame_id in range(1, keyframe_len + 1):
                query_id = frame_id * (-1)
                tmp_gt_depth = self.video.depths_gt[idx + query_id] 
                tmp_gt_c2w = self.video.poses_gt[idx + query_id]
                tmp_est_c2w = self.estimate_c2w_list[idx + query_id].cuda()
                warp_pose_batch += [tmp_est_c2w]
                warp_depth_img_batch += [tmp_gt_depth]
                tmp_gt_c2w = torch.from_numpy(pose_matrix_from_quaternion(tmp_gt_c2w.cpu())).cuda()
                warp_gt_pose_batch += [tmp_gt_c2w]
        
        if idx > 0:
            warp_depth_batch = torch.stack(warp_depth_img_batch, dim=0)
            warp_est_pose_batch = torch.stack(warp_pose_batch, dim=0)
            warp_gt_pose_batch = torch.stack(warp_gt_pose_batch, dim=0)
            intrinsic = np.array([self.fx, self.fy, self.cx, self.cy])
            batch_mask = depth_warp_pixel(cur_c2w, warp_est_pose_batch, cur_gt_depth.unsqueeze(-1), warp_depth_batch.unsqueeze(-1), cur_gt_depth,
                               intrinsic, H, W, batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_depth)
            batch_rays_o = batch_rays_o[batch_mask]
            batch_rays_d = batch_rays_d[batch_mask]
            batch_gt_depth = batch_gt_depth[batch_mask]
            batch_gt_color = batch_gt_color[batch_mask]
            i = i[batch_mask]
            j = j[batch_mask]
        
        if not color_refine:
            frame_pts_add = 0
            _ = self.gaussians.add_neural_points(batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color,
                                                 dynamic_radius=self.dynamic_r_add[
                                                     j, i] if self.use_dynamic_radius else None)
            frame_pts_add += _
            if self.pixels_based_on_color_grad > 0:
                batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, i, j = get_samples_with_pixel_grad(
                    0, H, 0, W, self.pixels_based_on_color_grad,
                    H, W, fx, fy, cx, cy, cur_c2w, gt_depth, gt_color, self.device,
                    depth_filter=True, return_index=True)
                _ = self.gaussians.add_neural_points(batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, \
                                                     is_pts_grad=True, dynamic_radius=self.dynamic_r_add[
                        j, i] if self.use_dynamic_radius else None)
                frame_pts_add += _

            if self.pixels_based_on_render and idx > 0:
                with torch.no_grad():
                    camera_center = cur_c2w[:3, 3]
                    world_view_transform = torch.inverse(cur_c2w).transpose(0, 1)
                    gaussians_opacity_activation = self.opacity_activation(self.gaussians_opacity)
                    gaussians_scaling_activation = self.scaling_activation(self.gaussians_scaling)
                    gaussians_rotation_activation = self.rotation_activation(self.gaussians_rotation)
                    render_pkg = render(self.gaussians_xyz, self.gaussians_features_dc, self.gaussians_features_rest, \
                                        gaussians_opacity_activation, gaussians_scaling_activation,
                                        gaussians_rotation_activation, \
                                        self.gaussians.get_active_sh_degree(), self.gaussians.get_max_sh_degree(),
                                        camera_center, world_view_transform, \
                                        self.projection_matrix, self.fovx, self.fovy, self.H, self.W)

                    image_render, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                        "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    depth_render = render_pkg["depth"][0]
                    image_render = image_render.permute(1, 2, 0)
                    opacity_render = render_pkg["acc"][0]
                    
                    batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, i, j = get_samples_point_add(
                        depth_render, image_render, opacity_render, 6000, self.add_pixel_depth_th, 0, H, 0, W,
                        H, W, fx, fy, cx, cy, cur_c2w, gt_depth, gt_color, self.device,
                        depth_filter=True, return_index=True, seg_mask = seg_mask)

                    keyframe_len = idx
                    warp_window = 3
                    warp_depth_img_batch = []
                    warp_pose_batch = []
                    warp_gt_pose_batch = []
                    if keyframe_len >= warp_window:
                        for frame_id in range(1, warp_window + 1):
                            query_id = frame_id * (-1)
                            tmp_gt_depth = self.video.depths_gt[idx + query_id]
                            tmp_gt_c2w = self.video.poses_gt[idx + query_id]
                            tmp_est_c2w = self.estimate_c2w_list[idx + query_id].cuda()
                            warp_pose_batch += [tmp_est_c2w]
                            warp_depth_img_batch += [tmp_gt_depth]
                            tmp_gt_c2w = torch.from_numpy(pose_matrix_from_quaternion(tmp_gt_c2w.cpu())).cuda()
                            warp_gt_pose_batch += [tmp_gt_c2w]
                    else:
                        for frame_id in range(1, keyframe_len + 1):
                            query_id = frame_id * (-1)
                            tmp_gt_depth = self.video.depths_gt[idx + query_id]                            
                            tmp_gt_c2w = self.video.poses_gt[idx + query_id]
                            tmp_est_c2w = self.estimate_c2w_list[idx + query_id].cuda()
                            warp_pose_batch += [tmp_est_c2w]
                            warp_depth_img_batch += [tmp_gt_depth]
                            tmp_gt_c2w = torch.from_numpy(pose_matrix_from_quaternion(tmp_gt_c2w.cpu())).cuda()
                            warp_gt_pose_batch += [tmp_gt_c2w]
                    if idx > 0:
                        warp_depth_batch = torch.stack(warp_depth_img_batch, dim=0)
                        warp_est_pose_batch = torch.stack(warp_pose_batch, dim=0)
                        warp_gt_pose_batch = torch.stack(warp_gt_pose_batch, dim=0)
                        intrinsic = np.array([self.fx, self.fy, self.cx, self.cy])
                        batch_mask = depth_warp_pixel(cur_c2w, warp_est_pose_batch, cur_gt_depth.unsqueeze(-1),
                                                      warp_depth_batch.unsqueeze(-1), cur_gt_depth,
                                                      intrinsic, H, W, batch_rays_o, batch_rays_d, batch_gt_depth,
                                                      batch_gt_depth)
                        batch_rays_o = batch_rays_o[batch_mask]
                        batch_rays_d = batch_rays_d[batch_mask]
                        batch_gt_depth = batch_gt_depth[batch_mask]
                        batch_gt_color = batch_gt_color[batch_mask]
                        i = i[batch_mask]
                        j = j[batch_mask]
                    
                    _ = self.gaussians.add_neural_points(batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color,
                                                         dynamic_radius=self.dynamic_r_add[
                                                             j, i] if self.use_dynamic_radius else None)
                    frame_pts_add += _

        self.gaussians_xyz = self.gaussians.get_xyz()
        self.gaussians_features_dc = self.gaussians.get_features_dc()
        self.gaussians_features_rest = self.gaussians.get_features_rest()
        self.gaussians_opacity = self.gaussians.get_opacity()
        self.gaussians_scaling = self.gaussians.get_scaling()
        self.gaussians_rotation = self.gaussians.get_rotation()

        masked_c_grad = {}
        mask_c2w = cur_c2w
        indices, indices_unseen = self.get_mask_from_c2w(mask_c2w, gt_depth_np)

        self.gaussians_xyz_grad = self.gaussians_xyz[indices].detach().clone().requires_grad_(True)
        self.gaussians_features_dc_grad = self.gaussians_features_dc[indices].detach().clone().requires_grad_(True)
        self.gaussians_features_rest_grad = self.gaussians_features_rest[indices].detach().clone().requires_grad_(True)
        self.gaussians_opacity_grad = self.gaussians_opacity[indices].detach().clone().requires_grad_(True)
        self.gaussians_scaling_grad = self.gaussians_scaling[indices].detach().clone().requires_grad_(True)
        self.gaussians_rotation_grad = self.gaussians_rotation[indices].detach().clone().requires_grad_(True)
        masked_c_grad['indices'] = indices

        if self.encode_exposure:
            mlp_exposure_para_list += list(self.mlp_exposure.parameters())

        optim_para_list = [
            {'params': [self.gaussians_xyz_grad], 'lr': self.position_lr_init, "name": "xyz"},
            {'params': [self.gaussians_features_dc_grad], 'lr': self.feature_lr, "name": "f_dc"},
            {'params': [self.gaussians_features_rest_grad], 'lr': self.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self.gaussians_opacity_grad], 'lr': self.opacity_lr, "name": "opacity"},
            {'params': [self.gaussians_scaling_grad], 'lr': self.scaling_lr, "name": "scaling"},
            {'params': [self.gaussians_rotation_grad], 'lr': self.rotation_lr, "name": "rotation"}
        ]

        if self.encode_exposure:
            optim_para_list.append(
                {'params': self.exposure_feat, 'lr': 0.001, "name": "expos_feat"})
            optim_para_list.append(
                {'params': mlp_exposure_para_list, 'lr': 0.005, "name": "mlp_expos_para"})

        self.optimizer = torch.optim.Adam(optim_para_list)

        if idx > 0 and not color_refine:
            num_joint_iters = 60
            num_joint_iters = np.clip(int(num_joint_iters * frame_pts_add / 5000), int(
                self.min_iter_ratio * num_joint_iters), 2 * num_joint_iters)
        total_gs_point_num = self.gaussians.get_xyz().shape[0]
        print('total gaussian points number: ', f'{total_gs_point_num}')

        actual_joint_iters = 0

        for joint_iter in range(num_joint_iters):
            if joint_iter <= (self.geo_iter_first if init else int(num_joint_iters * self.geo_iter_ratio)):
                self.stage = 'geometry'
            else:
                self.stage = 'color'

            self.optimizer.zero_grad()

            exposure_feat_list = []

            gt_depth = cur_gt_depth
            gt_color = cur_gt_color
            c2w = cur_c2w

            if self.encode_exposure:
                exposure_feat_list.append(self.exposure_feat)

            camera_center = c2w[:3, 3]
            world_view_transform = torch.inverse(c2w).transpose(0, 1)
            gaussians_opacity_grad_activation = self.opacity_activation(self.gaussians_opacity_grad)
            gaussians_scaling_grad_activation = self.scaling_activation(self.gaussians_scaling_grad)
            gaussians_rotation_grad_activation = self.rotation_activation(self.gaussians_rotation_grad)

            render_pkg = render(self.gaussians_xyz_grad, self.gaussians_features_dc_grad,
                                self.gaussians_features_rest_grad, \
                                gaussians_opacity_grad_activation, gaussians_scaling_grad_activation,
                                gaussians_rotation_grad_activation, \
                                self.gaussians.get_active_sh_degree(), self.gaussians.get_max_sh_degree(),
                                camera_center, world_view_transform, self.projection_matrix, \
                                self.fovx, self.fovy, self.H, self.W)

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            depth = render_pkg["depth"][0]
            image = image.permute(1, 2, 0)

            if self.encode_exposure:
                image = image.reshape(-1, 3)
                affine_tensor = self.mlp_exposure(exposure_feat_list[-1])
                rot, trans = affine_tensor[:9].reshape(3, 3), affine_tensor[-3:]
                image_slice = image.clone()
                image_slice = torch.matmul(image_slice, rot) + trans
                image = image_slice
                image = torch.sigmoid(image)
                image = image.reshape(self.H, self.W, 3)
            
            if joint_iter % 30 == 0:
                self.gaussians.oneupSHdegree()

            mask = (gt_depth > 0.0) & (gt_depth < 8.) & seg_mask
            mask = mask & (~torch.isnan(gt_depth))
            depths_wmask = depth[mask]
            gt_depths_wmask = gt_depth[mask]

            geo_loss = torch.abs(gt_depths_wmask - depths_wmask).sum()
            loss = geo_loss.clone() * self.w_geo_loss
            color_mask = mask
            color_loss = torch.abs(gt_color[color_mask] - image[color_mask]).sum()

            ssim_loss = (1.0 - ssim(image.permute(2, 0, 1), gt_color.permute(2, 0, 1).float()))
            weighted_ssim_loss = self.lambda_ssim_loss * ssim_loss
            loss += weighted_ssim_loss
            weighted_color_loss = self.w_color_loss * color_loss * (1 - self.lambda_ssim_loss)
            loss += weighted_color_loss

            loss.backward(retain_graph=False)
            self.max_radii2D = radii
            self.optimizer.step()
            self.optimizer.zero_grad()
            actual_joint_iters += 1

            total_gs_point_num_prune = self.gaussians_xyz.shape[0]
            if not self.wandb:
                if joint_iter % 100 == 0:
                    if self.stage == 'geometry':
                        print('iter: ', joint_iter, ', geo_loss: ', f'{geo_loss.item():0.6f}')
                    else:
                        print('iter: ', joint_iter, ', geo_loss: ', f'{geo_loss.item():0.6f}', ', color_loss: ', f'{color_loss.item():0.6f}')

            if joint_iter == num_joint_iters - 1:
                print('idx: ', idx.item(), ', geo_loss_pixel: ',
                      f'{(geo_loss.item() / mask.sum().item()):0.6f}',
                      ', color_loss_pixel: ', f'{(color_loss.item() / color_mask.sum().item()):0.4f}')
                if self.wandb:
                    wandb.log({'idx': int(idx.item()),
                                'geo_loss_pixel': float(f'{(geo_loss.item() / mask.sum().item()):0.6f}'),
                                'color_loss_pixel': float(f'{(color_loss.item() / color_mask.sum().item()):0.6f}'),
                                'pts_total': total_gs_point_num_prune})
                    wandb.log({'idx_map': int(idx.item()),
                               'num_joint_iters': num_joint_iters})

        if self.frustum_feature_selection:
            indices = masked_c_grad['indices']
            self.gaussians_xyz[indices] = self.gaussians_xyz_grad.detach().clone()
            self.gaussians_features_dc[indices] = self.gaussians_features_dc_grad.detach().clone()
            self.gaussians_features_rest[indices] = self.gaussians_features_rest_grad.detach().clone()
            self.gaussians_opacity[indices] = self.gaussians_opacity_grad.detach().clone()
            self.gaussians_scaling[indices] = self.gaussians_scaling_grad.detach().clone()
            self.gaussians_rotation[indices] = self.gaussians_rotation_grad.detach().clone()
        else:
            self.gaussians_xyz = self.gaussians_xyz_grad.detach().clone()
            self.gaussians_features_dc = self.gaussians_features_dc_grad.detach().clone()
            self.gaussians_features_rest = self.gaussians_features_rest_grad.detach().clone()
            self.gaussians_opacity = self.gaussians_opacity_grad.detach().clone()
            self.gaussians_scaling = self.gaussians_scaling_grad.detach().clone()
            self.gaussians_rotation = self.gaussians_rotation_grad.detach().clone()

        self.gaussians.update_xyz(self.gaussians_xyz.detach().clone())
        self.gaussians.update_features_dc(self.gaussians_features_dc.detach().clone())
        self.gaussians.update_features_rest(self.gaussians_features_rest.detach().clone())
        self.gaussians.update_scaling(self.gaussians_scaling.detach().clone())
        self.gaussians.update_rotation(self.gaussians_rotation.detach().clone())
        self.gaussians.update_opacity(self.gaussians_opacity.detach().clone())
        print('Current Map has been updated')

        if self.encode_exposure and idx == (self.n_img - 1):
            self.exposure_feat_all.append(self.exposure_feat.detach().cpu())
        return None

    def optimize_map(self, num_joint_iters, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w,
                     keyframe_dict, keyframe_list, cur_c2w, color_refine=False, cur_seg_mask = None):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enables).

        Args:
            num_joint_iters (int): number of mapping iterations.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            keyframe_dict (list): list of keyframes info dictionary.
            keyframe_list (list): list of keyframe index.
            cur_c2w (tensor): the estimated camera to world matrix of current frame. 
            color_refine (bool): whether to do color refinement (optimize color features with fixed color decoder).

        Returns:
            cur_c2w/None (tensor/None): return the updated cur_c2w, return None if no BA
        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        cfg = self.cfg
        device = self.device
        init = True if idx == 0 else False
        bottom = torch.tensor([0, 0, 0, 1.0], device=self.device).reshape(1, 4)

        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                num = self.mapping_window_size-2
                optimize_frame = random_select(len(self.keyframe_dict)-1, num)
            elif self.keyframe_selection_method == 'overlap':
                num = self.mapping_window_size-2
                optimize_frame = self.keyframe_selection_overlap(
                    cur_gt_color, cur_gt_depth, cur_c2w, keyframe_dict[:-1], num)

        oldest_frame = None
        if len(keyframe_list) > 0:
            optimize_frame = optimize_frame + [len(keyframe_list)-1]
            oldest_frame = min(optimize_frame)
        optimize_frame += [-1]

        opt_frames_camera_tensor_list = []
        opt_frames_gt_depth_list = []

        if self.save_selected_keyframes_info:
            keyframes_info = []
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    frame_idx = keyframe_list[frame]
                    tmp_gt_c2w = keyframe_dict[frame]['gt_c2w']
                    tmp_est_c2w = keyframe_dict[frame]['est_c2w']
                    tmp_gt_depth = keyframe_dict[frame]['depth']
                else:
                    frame_idx = idx
                    tmp_gt_c2w = gt_cur_c2w
                    tmp_est_c2w = cur_c2w
                    tmp_gt_depth = cur_gt_depth
                keyframes_info.append(
                    {'idx': frame_idx, 'gt_c2w': tmp_gt_c2w, 'est_c2w': tmp_est_c2w})
                opt_frames_camera_tensor_list.append(tmp_est_c2w)
                opt_frames_gt_depth_list.append(tmp_gt_depth.cpu().numpy())
            self.selected_keyframes[idx] = keyframes_info

        mlp_exposure_para_list = []
        self.gaussians_xyz = self.gaussians.get_xyz()
        self.gaussians_features_dc = self.gaussians.get_features_dc()
        self.gaussians_features_rest = self.gaussians.get_features_rest()
        self.gaussians_opacity = self.gaussians.get_opacity()
        self.gaussians_scaling = self.gaussians.get_scaling()
        self.gaussians_rotation = self.gaussians.get_rotation()

        indices = None
        indices_unseen = None
        if self.frustum_feature_selection:  # required if not color_refine
            masked_c_grad = {}
            for i in range(len(opt_frames_camera_tensor_list)):
                mask_c2w = opt_frames_camera_tensor_list[i]
                mask_depth_np = opt_frames_gt_depth_list[i]
                indices_i, indices_us_i = self.get_mask_from_c2w(mask_c2w, mask_depth_np)
                indices = list(set(indices) | (set(indices_i))) if indices is not None else indices_i
                indices_unseen = list(set(indices_unseen) & (set(indices_us_i))) if indices_unseen is not None else indices_us_i

            gaussians_xyz_unfrustum = self.gaussians_xyz[indices_unseen].detach().clone()
            gaussians_features_dc_unfrustum = self.gaussians_features_dc[indices_unseen].detach().clone()
            gaussians_features_rest_unfrustum = self.gaussians_features_rest[indices_unseen].detach().clone()
            gaussians_opacity_unfrustum = self.gaussians_opacity[indices_unseen].detach().clone()
            gaussians_scaling_unfrustum = self.gaussians_scaling[indices_unseen].detach().clone()
            gaussians_rotation_unfrustum = self.gaussians_rotation[indices_unseen].detach().clone()

            self.gaussians_xyz_grad = self.gaussians_xyz[indices].detach().clone().requires_grad_(True)
            self.gaussians_features_dc_grad = self.gaussians_features_dc[indices].detach().clone().requires_grad_(True)
            self.gaussians_features_rest_grad = self.gaussians_features_rest[indices].detach().clone().requires_grad_(True)
            self.gaussians_opacity_grad = self.gaussians_opacity[indices].detach().clone().requires_grad_(True)
            self.gaussians_scaling_grad = self.gaussians_scaling[indices].detach().clone().requires_grad_(True)
            self.gaussians_rotation_grad = self.gaussians_rotation[indices].detach().clone().requires_grad_(True)
            masked_c_grad['indices'] = indices
        else:
            masked_c_grad = {}
            self.gaussians_xyz_grad = self.gaussians_xyz.detach().clone().requires_grad_(True)
            self.gaussians_features_dc_grad = self.gaussians_features_dc.detach().clone().requires_grad_(True)
            self.gaussians_features_rest_grad = self.gaussians_features_rest.detach().clone().requires_grad_(True)
            self.gaussians_opacity_grad = self.gaussians_opacity.detach().clone().requires_grad_(True)
            self.gaussians_scaling_grad = self.gaussians_scaling.detach().clone().requires_grad_(True)
            self.gaussians_rotation_grad = self.gaussians_rotation.detach().clone().requires_grad_(True)

        if self.encode_exposure:
            mlp_exposure_para_list += list(self.mlp_exposure.parameters())

        if self.BA:
            camera_tensor_list = []
            gt_camera_tensor_list = []
            for frame in optimize_frame:
                if frame != oldest_frame:
                    if frame != -1:
                        c2w = keyframe_dict[frame]['est_c2w']
                        gt_c2w = keyframe_dict[frame]['gt_c2w']
                    else:
                        c2w = cur_c2w
                        gt_c2w = gt_cur_c2w
                    camera_tensor = get_tensor_from_camera(c2w)
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    camera_tensor_list.append(camera_tensor)
                    gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                    gt_camera_tensor_list.append(gt_camera_tensor)
        
        optim_para_list = [
            {'params': [self.gaussians_xyz_grad], 'lr': self.position_lr_init, "name": "xyz"},
            {'params': [self.gaussians_features_dc_grad], 'lr': self.feature_lr, "name": "f_dc"},
            {'params': [self.gaussians_features_rest_grad], 'lr': self.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self.gaussians_opacity_grad], 'lr': self.opacity_lr, "name": "opacity"},
            {'params': [self.gaussians_scaling_grad], 'lr': self.scaling_lr, "name": "scaling"},
            {'params': [self.gaussians_rotation_grad], 'lr': self.rotation_lr, "name": "rotation"}
        ]

        if self.BA:
            optim_para_list.append({'params': camera_tensor_list, 'lr': 0, "name": "cam_tensor"})

        if self.encode_exposure:
            optim_para_list.append(
                {'params': self.exposure_feat, 'lr': 0.001, "name": "expos_feat"})
            optim_para_list.append(
                {'params': mlp_exposure_para_list, 'lr': 0.005, "name": "mlp_expos_para"})

        self.optimizer = torch.optim.Adam(optim_para_list)
        if idx > 0 and not color_refine:
            num_joint_iters = 100

        total_gs_point_num = self.gaussians.get_xyz().shape[0]
        print('total gaussian points number: ', f'{total_gs_point_num}')

        if self.frustum_feature_selection:
            frustum_gs_point_num = len(indices)
            print('cur window points number: ', f'{frustum_gs_point_num}')
            unseen_gs_point_num = len(indices_unseen)
            print('cur unseen window points number: ', f'{unseen_gs_point_num}')

        actual_joint_iters = 0

        for joint_iter in range(num_joint_iters):
            if joint_iter <= (self.geo_iter_first if init else int(num_joint_iters*self.geo_iter_ratio)):
                self.stage = 'geometry' 
            else:
                self.stage = 'color' 

            if self.BA:
                if joint_iter >= num_joint_iters*(self.geo_iter_ratio+0.2) and (joint_iter <= num_joint_iters*(self.geo_iter_ratio+0.6)):
                    self.optimizer.param_groups[6]['lr'] = self.BA_cam_lr
                else:
                    self.optimizer.param_groups[6]['lr'] = 0.0
            
            self.optimizer.zero_grad()

            images = []
            depths = []
            gt_colors = []
            gt_depths = []
            exposure_feat_list = []
            seg_masks = []
            camera_tensor_id = 0

            for frame in optimize_frame:
                if frame != -1:
                    gt_depth = keyframe_dict[frame]['depth']
                    gt_color = keyframe_dict[frame]['color']
                    seg_mask = keyframe_dict[frame]["seg_mask"]
                    if self.BA and frame != oldest_frame:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        camera_tensor_id += 1
                        c2w = get_camera_from_tensor(camera_tensor)
                        c2w = torch.cat([c2w, bottom], dim=0)
                    else:
                        c2w = keyframe_dict[frame]['est_c2w']
                else:
                    gt_depth = cur_gt_depth
                    gt_color = cur_gt_color
                    seg_mask = cur_seg_mask
                    if self.BA:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        c2w = get_camera_from_tensor(camera_tensor)
                        c2w = torch.cat([c2w, bottom], dim=0)
                    else:
                        c2w = cur_c2w

                if self.encode_exposure:
                    exposure_feat_list.append(
                        self.exposure_feat if frame == -1 else keyframe_dict[frame]['exposure_feat'])
                
                camera_center = c2w[:3,3]
                world_view_transform = torch.inverse(c2w).transpose(0,1)
                gaussians_opacity_grad_activation = self.opacity_activation(self.gaussians_opacity_grad)
                gaussians_scaling_grad_activation = self.scaling_activation(self.gaussians_scaling_grad)
                gaussians_rotation_grad_activation = self.rotation_activation(self.gaussians_rotation_grad)

                render_pkg = render(self.gaussians_xyz_grad, self.gaussians_features_dc_grad, self.gaussians_features_rest_grad,\
                                     gaussians_opacity_grad_activation, gaussians_scaling_grad_activation, gaussians_rotation_grad_activation,\
                                        self.gaussians.get_active_sh_degree(), self.gaussians.get_max_sh_degree(), camera_center, world_view_transform, self.projection_matrix,\
                                              self.fovx, self.fovy, self.H, self.W)

                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                depth = render_pkg["depth"][0]
                image = image.permute(1,2,0)
                
                if self.encode_exposure:
                    image = image.reshape(-1,3)
                    affine_tensor = self.mlp_exposure(exposure_feat_list[-1])
                    rot, trans = affine_tensor[:9].reshape(3, 3), affine_tensor[-3:]
                    image_slice = image.clone()
                    image_slice = torch.matmul(image_slice, rot) + trans
                    image = image_slice 
                    image = torch.sigmoid(image)
                    image = image.reshape(self.H, self.W, 3)

                images.append(image.unsqueeze(0))
                depths.append(depth.unsqueeze(0))
                gt_colors.append(gt_color.unsqueeze(0))
                gt_depths.append(gt_depth.unsqueeze(0))
                seg_masks.append(seg_mask.unsqueeze(0))

            if joint_iter % 30 == 0:
                self.gaussians.oneupSHdegree()

            images = torch.cat(images)
            depths = torch.cat(depths)
            gt_colors = torch.cat(gt_colors)
            gt_depths = torch.cat(gt_depths)
            seg_masks = torch.cat(seg_masks)

            mask = (gt_depths > 0.0) & (gt_depths < 8.) & seg_masks
            mask = mask & (~torch.isnan(gt_depths))
            depths_wmask = depths[mask]
            gt_depths_wmask = gt_depths[mask]

            geo_loss = torch.abs(gt_depths_wmask-depths_wmask).sum()
            loss = geo_loss.clone() * self.w_geo_loss
            color_mask = mask
            color_loss = torch.abs(gt_colors[color_mask] - images[color_mask]).sum()

            for i in range(images.shape[0]):
                image = images[i]
                gt_color = gt_colors[i]
                ssim_loss = (1.0 - ssim(image.permute(2, 0, 1), gt_color.permute(2, 0, 1).float()))
                weighted_ssim_loss = self.lambda_ssim_loss*ssim_loss
                loss += weighted_ssim_loss
            weighted_color_loss = self.w_color_loss*color_loss*(1-self.lambda_ssim_loss)
            loss += weighted_color_loss

            loss.backward(retain_graph=False)

            self.max_radii2D = radii
            with torch.no_grad():
                if joint_iter > 0 and joint_iter % 20 == 0:
                    self.prune_neural_point(0.001)

                
                self.optimizer.step()
                self.optimizer.zero_grad()
                actual_joint_iters += 1
            
            total_gs_point_num_prune = self.gaussians_xyz.shape[0]
            if not self.wandb:
                if joint_iter % 100 == 0:
                    if self.stage == 'geometry':
                        print('iter: ', joint_iter, ', geo_loss: ', f'{geo_loss.item():0.6f}')
                    else:
                        print('iter: ', joint_iter, ', geo_loss: ', f'{geo_loss.item():0.6f}', ', color_loss: ', f'{color_loss.item():0.6f}')

            if joint_iter == num_joint_iters-1:
                print('idx: ', idx.item(), ', geo_loss_pixel: ', f'{(geo_loss.item()/mask.sum().item()):0.6f}',
                      ', color_loss_pixel: ', f'{(color_loss.item()/color_mask.sum().item()):0.4f}')
                if self.wandb:
                    wandb.log({'idx': int(idx.item()),
                                'geo_loss_pixel': float(f'{(geo_loss.item()/mask.sum().item()):0.6f}'),
                                'color_loss_pixel': float(f'{(color_loss.item()/color_mask.sum().item()):0.6f}'),
                                'pts_total': total_gs_point_num_prune})

                    wandb.log({'idx_map': int(idx.item()),
                               'num_joint_iters': num_joint_iters})
                    
        if self.frustum_feature_selection:
            indices = masked_c_grad['indices']
            if len(indices_unseen) == 0:
                self.gaussians_xyz[indices] = self.gaussians_xyz_grad.detach().clone()
                self.gaussians_features_dc[indices] = self.gaussians_features_dc_grad.detach().clone()
                self.gaussians_features_rest[indices] = self.gaussians_features_rest_grad.detach().clone()
                self.gaussians_opacity[indices] = self.gaussians_opacity_grad.detach().clone()
                self.gaussians_scaling[indices] = self.gaussians_scaling_grad.detach().clone()
                self.gaussians_rotation[indices] = self.gaussians_rotation_grad.detach().clone()
            else:
                self.gaussians_xyz = torch.cat((self.gaussians_xyz_grad.detach().clone(), gaussians_xyz_unfrustum.detach().clone()), 0)
                self.gaussians_features_dc = torch.cat((self.gaussians_features_dc_grad.detach().clone(), gaussians_features_dc_unfrustum.detach().clone()), 0)
                self.gaussians_features_rest = torch.cat((self.gaussians_features_rest_grad.detach().clone(), gaussians_features_rest_unfrustum.detach().clone()), 0)
                self.gaussians_opacity = torch.cat((self.gaussians_opacity_grad.detach().clone(), gaussians_opacity_unfrustum.detach().clone()), 0)
                self.gaussians_scaling = torch.cat((self.gaussians_scaling_grad.detach().clone(), gaussians_scaling_unfrustum.detach().clone()), 0)
                self.gaussians_rotation = torch.cat((self.gaussians_rotation_grad.detach().clone(), gaussians_rotation_unfrustum.detach().clone()), 0)
                masked_c_grad['indices'] = np.arange(self.gaussians_xyz_grad.shape[0]).tolist()
                indices_unseen = np.arange(self.gaussians_xyz_grad.shape[0], self.gaussians_xyz_grad.shape[0] + gaussians_xyz_unfrustum.shape[0]).tolist()
        else:
            self.gaussians_xyz = self.gaussians_xyz_grad.detach().clone()
            self.gaussians_features_dc = self.gaussians_features_dc_grad.detach().clone()
            self.gaussians_features_rest = self.gaussians_features_rest_grad.detach().clone()
            self.gaussians_opacity = self.gaussians_opacity_grad.detach().clone()
            self.gaussians_scaling = self.gaussians_scaling_grad.detach().clone()
            self.gaussians_rotation = self.gaussians_rotation_grad.detach().clone()

        if self.frustum_feature_selection:
            self.gaussians.update_xyz(self.gaussians_xyz.detach().clone())
            self.gaussians.update_features_dc(self.gaussians_features_dc.detach().clone())
            self.gaussians.update_features_rest(self.gaussians_features_rest.detach().clone())
            self.gaussians.update_scaling(self.gaussians_scaling.detach().clone())
            self.gaussians.update_rotation(self.gaussians_rotation.detach().clone())
            self.gaussians.update_opacity(self.gaussians_opacity.detach().clone())
        else:
            self.gaussians.update_xyz(self.gaussians_xyz.detach().clone())
            self.gaussians.update_features_dc(self.gaussians_features_dc.detach().clone())
            self.gaussians.update_features_rest(self.gaussians_features_rest.detach().clone())
            self.gaussians.update_scaling(self.gaussians_scaling.detach().clone())
            self.gaussians.update_rotation(self.gaussians_rotation.detach().clone())
            self.gaussians.update_opacity(self.gaussians_opacity.detach().clone())
        print('Mapper has updated point features.')

        if self.BA:
            camera_tensor_id = 0
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    if frame != oldest_frame:
                        c2w = get_camera_from_tensor(
                            camera_tensor_list[camera_tensor_id].detach())
                        c2w = torch.cat([c2w, bottom], dim=0)
                        camera_tensor_id += 1
                        keyframe_dict[frame]['est_c2w'] = c2w.clone()
                else:
                    c2w = get_camera_from_tensor(
                        camera_tensor_list[-1].detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                    cur_c2w = c2w.clone()
            print('Mapper has updated optimize pose (BA).')

        if self.encode_exposure and idx == (self.n_img-1):
            self.exposure_feat_all.append(self.exposure_feat.detach().cpu())
        if self.BA:
            return cur_c2w
        else:
            return None

    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.

        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            self.gaussians_xyz = self.gaussians.get_xyz().detach().clone()
            self.gaussians_features_dc = self.gaussians.get_features_dc().detach().clone()
            self.gaussians_features_rest = self.gaussians.get_features_rest().detach().clone()
            self.gaussians_opacity = self.gaussians.get_opacity().detach().clone()
            self.gaussians_scaling = self.gaussians.get_scaling().detach().clone()
            self.gaussians_rotation = self.gaussians.get_rotation().detach().clone()
            self.prev_mapping_idx = self.mapping_idx[0].clone()
            if self.verbose:
                print('Tracker has updated the parameters from Mapper.')

    def optimize_cam_in_batch(self, camera_tensor, gt_color, gt_depth, batch_size, optimizer, seg_mask,
                              selected_index=None):
        """
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            camera_tensor (tensor): camera tensor.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.
            selected_index: top color gradients pixels are pre-selected.

        Returns:
            loss (float): total loss
            color_loss (float): color loss component
            geo_loss (float): geometric loss component
        """
        H, W = self.H, self.W
        optimizer.zero_grad()
        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H

        with torch.no_grad():
            inside_mask = gt_depth <= torch.minimum(
                10*gt_depth.median(), 1.2*torch.max(gt_depth))
            edge_mask = torch.zeros((H, W), dtype=torch.bool)
            edge_mask = edge_mask.to(inside_mask)
            edge_mask[Hedge:H-Hedge, Wedge:W-Wedge] = True

        camera_center = c2w[:3,3]
        world_view_transform = torch.inverse(convert3x4_4x4(c2w)).transpose(0,1)
        gaussians_opacity_activation = self.opacity_activation(self.gaussians_opacity)
        gaussians_scaling_activation = self.scaling_activation(self.gaussians_scaling)
        gaussians_rotation_activation = self.rotation_activation(self.gaussians_rotation)
        render_pkg = render(self.gaussians_xyz, self.gaussians_features_dc, self.gaussians_features_rest,\
                                gaussians_opacity_activation, gaussians_scaling_activation, gaussians_rotation_activation,\
                                    self.gaussians.get_active_sh_degree(), self.gaussians.get_max_sh_degree(), camera_center, world_view_transform,\
                                        self.projection_matrix, self.fovx, self.fovy, H, W)
       
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth = render_pkg["depth"][0]
        image = image.permute(1,2,0)
        opacity_acc = render_pkg["acc"]
        opacity_mask = opacity_acc > self.opacity_thres
        opacity_mask = opacity_mask.squeeze()

        if self.encode_exposure:
            image = image.reshape(-1,3)
            affine_tensor = self.mlp_exposure(self.exposure_feat)
            rot, trans = affine_tensor[:9].reshape(3, 3), affine_tensor[-3:]
            image_slice = image.clone()
            image_slice = torch.matmul(image_slice, rot) + trans
            image = image_slice
            image = torch.sigmoid(image)
            image = image.reshape(self.H, self.W, 3)

        nan_mask = (~torch.isnan(depth))
        if self.ignore_outlier_depth_loss:
            depth_error = torch.abs(gt_depth-depth) * (gt_depth > 0)
            mask = (depth_error < 10*(depth_error[opacity_mask].median()))
            mask = mask & (depth > 0) & (gt_depth > 0) & (gt_depth < 8.0)
        else:
            mask = (gt_depth > 0) & (gt_depth < 8.0)

        mask = mask & nan_mask
        mask = mask & edge_mask
        mask = mask & seg_mask

        if self.use_opacity_mask_for_loss:
            mask = mask & opacity_mask

        geo_loss = torch.clamp((torch.abs(gt_depth-depth)), min=0.0, max=1e3)[mask].sum()
        loss = self.w_geo_loss_tracking * geo_loss
        color_loss = torch.abs(gt_color - image)[mask].sum()
        if self.use_color_in_tracking:
            loss += self.w_color_loss_tracking*color_loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.opacity_mask = opacity_mask
        return loss.item(), (color_loss/mask.shape[0]).item(), (geo_loss/mask.shape[0]).item(), image, depth, mask
    
    def convert_relative_pose(self, idx):
        poses = torch.zeros((idx+1, 4, 4))
        for i in range(idx+1):
            if i % self.keyframe_every == 0:
                poses[i] = self.estimate_c2w_rel_list[i]
            else:
                kf_id = i // self.keyframe_every
                kf_frame_id = kf_id * self.keyframe_every
                c2w_key = self.estimate_keyframe_dict[kf_frame_id]
                delta = self.estimate_c2w_rel_list[i]
                poses[i] = delta @ c2w_key
        return poses

    def tracking(self, idx, gt_color, gt_depth, gt_c2w, estimated_c2w, seg_mask):
        device = self.device
        self.update_para_from_mapping()

        if self.verbose:
            print(Fore.MAGENTA)
            print("Tracking KeyFrame ",  idx.item())
            print(Style.RESET_ALL)

        if idx <= 1:
            c2w = gt_c2w
        else:
            self.count_bound = (idx % 5) if (idx % 5) > 0 else 5
            gt_camera_tensor = get_tensor_from_camera(gt_c2w)
            self.num_cam_iters = self.cfg['tracking']['iters']
            estimated_new_cam_c2w = estimated_c2w
            camera_tensor = get_tensor_from_camera(
                estimated_new_cam_c2w.detach())
            
            if torch.dot(camera_tensor[:4], gt_camera_tensor[:4]).item() < 0:
                camera_tensor[:4] *= -1

            if self.separate_LR:
                camera_tensor = camera_tensor.to(device).detach()
                T = camera_tensor[-3:]
                quad = camera_tensor[:4]
                self.quad = Variable(quad, requires_grad=True)
                self.T = Variable(T, requires_grad=True)
                camera_tensor = torch.cat([quad, T], 0)
                cam_para_list_T = [self.T]
                cam_para_list_quad = [self.quad]
                optim_para_list = [{'params': cam_para_list_T, 'lr': self.cam_lr},
                                    {'params': cam_para_list_quad, 'lr': self.cam_lr*0.4}]
            else:
                camera_tensor = Variable(
                    camera_tensor.to(device), requires_grad=True)
                cam_para_list = [camera_tensor]
                optim_para_list = [
                    {'params': cam_para_list, 'lr': self.cam_lr}]
                
            optimizer_camera = torch.optim.Adam(optim_para_list)

            candidate_cam_tensor = None
            current_min_loss = float(1e20)

            actual_cam_iters = 0
            for cam_iter in range(self.num_cam_iters):
                actual_cam_iters +=1
                if self.separate_LR:
                    camera_tensor = torch.cat(
                        [self.quad, self.T], 0).to(self.device)

                loss, color_loss_pixel, geo_loss_pixel, image, depth, mask =\
                        self.optimize_cam_in_batch(camera_tensor, gt_color, gt_depth, self.tracking_pixels, optimizer_camera, seg_mask)

                if cam_iter == 0:
                    initial_loss = loss

                loss_camera_tensor = torch.abs(
                    gt_camera_tensor.to(device)-camera_tensor).mean().item()
                if loss <= current_min_loss:
                    current_min_loss = loss
                    candidate_cam_tensor = camera_tensor.detach().clone()
                if cam_iter == self.num_cam_iters-1:
                    if not self.wandb:
                        print(f'idx:{idx}, re-rendering loss: {initial_loss:.2f}->{current_min_loss:.2f}.')
               
                if (cam_iter + 1) % 20 == 0 or cam_iter == 0:
                    if not self.wandb:
                        print(f'iter: {cam_iter}, camera tensor error: {loss_camera_tensor:.4f},  tracking loss: {loss:.4f}')

            bottom = torch.tensor(
                [0, 0, 0, 1.0], device=self.device).reshape(1, 4)
            c2w = get_camera_from_tensor(
                candidate_cam_tensor.detach().clone())
            c2w = torch.cat([c2w, bottom], dim=0)
            print(f'Finish tracking opt, plan iter: {self.num_cam_iters}, actual opt iter: {actual_cam_iters}')

        self.estimate_c2w_list[idx] = c2w.clone().cpu()
        self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
        self.pre_c2w = c2w.clone()
        self.idx[0] = idx

        if self.low_gpu_mem:
            torch.cuda.empty_cache()
        if self.wandb:
            wandb.finish()

    def mapping(self, idx, gt_color, gt_depth, gt_c2w, estimated_c2w, seg_mask, init = False):        
        if self.verbose:
            print(Fore.GREEN)
            print("Mapping KeyFrame ", idx.item())
            print(Style.RESET_ALL)
        
        color_refine = True if (idx == self.n_img-1 and self.color_refine) else False

        if not init:                                                                                                                                                                 
            num_joint_iters = self.cfg['mapping']['iters']
            self.mapping_window_size = self.cfg['mapping']['mapping_window_size']*(
                2 if self.n_img > 8000 else 1)
            if idx == self.n_img-1 and self.color_refine:
                outer_joint_iters = 1
                self.mapping_window_size *= 2
                self.geo_iter_ratio = 0.0
                num_joint_iters *= 2
                self.fix_color_decoder = True
                self.frustum_feature_selection = False
                self.keyframe_selection_method = 'global'
            else:
                outer_joint_iters = 1
        else:
            outer_joint_iters = 1
            num_joint_iters = self.iters_first

        cur_c2w = self.estimate_c2w_list[idx].to(self.device)
        for outer_joint_iter in range(outer_joint_iters):
            self.BA = (len(self.keyframe_list) >
                        4) and self.cfg['mapping']['BA']
            self.optimize_cur_map(num_joint_iters, idx, gt_color, gt_depth, gt_c2w, cur_c2w, color_refine=color_refine, seg_mask = seg_mask)
            _ = self.optimize_map(num_joint_iters, idx, gt_color, gt_depth, gt_c2w,
                                    self.keyframe_dict, self.keyframe_list, cur_c2w, color_refine=color_refine, cur_seg_mask = seg_mask)
            if self.BA:
                cur_c2w = _
                self.estimate_c2w_list[idx] = cur_c2w

        if (idx % self.keyframe_every == 0 or (idx == self.n_img-2)) and (idx not in self.keyframe_list) and (not torch.isinf(gt_c2w).any()) and (not torch.isnan(gt_c2w).any()):
            self.keyframe_list.append(idx)
            dic_of_cur_frame = {'gt_c2w': gt_c2w.detach(), 'idx': idx, 'color': gt_color.detach(),
                                'depth': gt_depth.detach(), 'est_c2w': cur_c2w.detach().clone(), "seg_mask": seg_mask.detach()}
            if self.use_dynamic_radius:
                dic_of_cur_frame.update(
                    {'dynamic_r_query': self.dynamic_r_query.detach()})
            if self.encode_exposure:
                dic_of_cur_frame.update(
                    {'exposure_feat': self.exposure_feat.detach()})
            self.keyframe_dict.append(dic_of_cur_frame)
        
        self.pre_c2w = self.estimate_c2w_list[idx].to(self.device)

        if (idx > 0 and idx % self.ckpt_freq == 0) or idx == self.n_img-1:
            self.logger.log(idx, self.keyframe_dict, self.keyframe_list,
                            selected_keyframes=self.selected_keyframes
                            if self.save_selected_keyframes_info else None, gaussians=self.gaussians,
                            exposure_feat=self.exposure_feat_all
                            if self.encode_exposure else None)
        if idx == self.n_img-1:
            print('Color refinement done.')
            print('Mapper finished.')
        if self.low_gpu_mem:
            torch.cuda.empty_cache() 

    def run(self, idx, gt_color, gt_depth, gt_c2w, estimated_c2w, seg_mask): 
        """
        Dispatch Threads. # this func, when called, act as main process
        """
        gt_color = gt_color.permute(1,2,0)

        if self.edge is not None:
            seg_mask = seg_mask[self.edge:-self.edge, self.edge:-self.edge]
            gt_color = gt_color[self.edge:-self.edge, self.edge:-self.edge]
            gt_depth = gt_depth[self.edge:-self.edge, self.edge:-self.edge]
        if idx == 0:
            init = True
        else:
            init = False

        if self.use_dynamic_radius:
            ratio = self.radius_query_ratio
            intensity = rgb2gray(gt_color.cpu().numpy())
            grad_y = filters.sobel_h(intensity)
            grad_x = filters.sobel_v(intensity)
            color_grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            color_grad_mag = np.clip(
                color_grad_mag, 0.0, self.color_grad_threshold)
            if idx > 0:
                fn_map_r_add = interp1d([0, 0.01, self.color_grad_threshold], [
                    self.radius_add_max, self.radius_add_max, self.radius_add_min])
                fn_map_r_query = interp1d([0, 0.01, self.color_grad_threshold], [
                    ratio * self.radius_add_max, ratio * self.radius_add_max, ratio * self.radius_add_min])
            else:
                fn_map_r_add = interp1d([0, 0.01, self.color_grad_threshold], [
                    self.radius_add_max / 2, self.radius_add_max / 2, self.radius_add_min / 2])
                fn_map_r_query = interp1d([0, 0.01, self.color_grad_threshold], [ratio * self.radius_add_max / 2, ratio * self.radius_add_max / 2,
                    ratio * self.radius_add_min / 2])

            dynamic_r_add = fn_map_r_add(color_grad_mag)
            dynamic_r_query = fn_map_r_query(color_grad_mag)
            self.dynamic_r_add, self.dynamic_r_query = torch.from_numpy(dynamic_r_add).to(
                self.device), torch.from_numpy(dynamic_r_query).to(self.device)
        
        if idx == 0:
            self.estimate_c2w_list[0] = gt_c2w.cpu()
            self.gt_c2w_list[0] = gt_c2w.cpu()
            self.mapping(idx, gt_color, gt_depth, gt_c2w, estimated_c2w, seg_mask,  init)
            self.tracking(idx, gt_color, gt_depth, gt_c2w, estimated_c2w, seg_mask)

        else:
            self.tracking(idx, gt_color, gt_depth, gt_c2w, estimated_c2w, seg_mask)
            self.mapping(idx, gt_color, gt_depth, gt_c2w, estimated_c2w, seg_mask, init)
