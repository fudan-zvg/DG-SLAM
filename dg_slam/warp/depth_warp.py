import torch
from warp.utils import get_samples_by_indices_batch, project_point3d_to_image_batch

def depth_warp_to_mask(init_pose, last_pose, images, last_image, depths, intrinsics, H, W, threshold = 0.3):
    last_pose[:,:3,1:3] = last_pose[:,:3,1:3] * (-1) 
    init_pose[:,:3,1:3] = init_pose[:,:3,1:3] * (-1)

    patch_size = 1
    fx, fy, cx, cy = intrinsics[0].item(), intrinsics[1].item(), intrinsics[2].item(), intrinsics[3].item()

    patch_sample_mask = torch.ones_like(depths).squeeze(0)
    batch_gt_uv = torch.nonzero(patch_sample_mask).float().cuda()
    batch_gt_uv = batch_gt_uv.flip(dims=[1])
    expand_current_c2w = init_pose.repeat(batch_gt_uv.shape[0], 1, 1).cuda()

    batch_patch_uv = (batch_gt_uv.clone().view(1, *batch_gt_uv.shape) )  

    batch_patch_uv[:, :, 0] = batch_patch_uv[:, :, 0] / W * 2 - 1.0
    batch_patch_uv[:, :, 1] = batch_patch_uv[:, :, 1] / H * 2 - 1.0

    batch_patch_rays_o, batch_patch_rays_d, batch_patch_gt_depth, batch_patch_gt_color = get_samples_by_indices_batch(
        0, H, 0, W, H, W, fx, fy, cx, cy, 
        expand_current_c2w, depths, images, batch_patch_uv, batch_patch_uv.device, )

    channel = images.shape[-1]
    batch_patch_gt_depth = batch_patch_gt_depth.view( patch_size *  patch_size, -1 )
    batch_patch_gt_color = batch_patch_gt_color.view( patch_size *  patch_size, -1, channel)

    patch_3d_pts = (batch_patch_rays_o + batch_patch_rays_d * batch_patch_gt_depth[:, :, None]).float()
    uv, z = project_point3d_to_image_batch(
            last_pose, patch_3d_pts.view(-1, 3, 1), fx, fy, cx, cy, patch_3d_pts.device)
    edge = 0
    uv = uv.view(patch_3d_pts.shape[0], patch_3d_pts.shape[1], last_pose.shape[0], 2) 
    mask = ((uv[(patch_size * patch_size) // 2, :, :, 0] < W - edge)
            * (uv[(patch_size * patch_size) // 2, :, :, 0] > edge)
            * (uv[(patch_size * patch_size) // 2, :, :, 1] < H - edge)
            * (uv[(patch_size * patch_size) // 2, :, :, 1] > edge))
    mask = mask & (z.view(patch_3d_pts.shape[0], patch_3d_pts.shape[1], last_pose.shape[0], 1
            )[(patch_size * patch_size) // 2, :, :, 0]<= 0)

    windows_reproj_idx = uv.permute(2, 1, 0, 3)
    windows_reproj_idx[..., 0] = windows_reproj_idx[..., 0] / W * 2.0 - 1.0
    windows_reproj_idx[..., 1] = windows_reproj_idx[..., 1] / H * 2.0 - 1.0

    windows_reproj_gt_color = torch.nn.functional.grid_sample(last_image.permute(0, 3, 1, 2).float(),
        windows_reproj_idx, padding_mode="border",).permute( 2, 0, 3, 1 )
    windows_reproj_gt_color = windows_reproj_gt_color.squeeze(1)
    windows_reproj_gt_color = windows_reproj_gt_color.squeeze(1)
    batch_gt_color = batch_patch_gt_color.permute(1, 0, 2).squeeze(1)

    mask = mask.view(H, W)
    batch_gt_color = batch_gt_color.view(H, W,channel)
    windows_reproj_gt_color = windows_reproj_gt_color.view(H, W,channel)

    color_residual = torch.abs(windows_reproj_gt_color - batch_gt_color)
    color_residual = torch.mean(color_residual, dim = -1)
    
    warp_mask = (color_residual < threshold)
    return warp_mask.unsqueeze(0)

def depth_warp_pixel(init_pose, last_pose, images, last_image, depths, intrinsics, H, W, batch_patch_rays_o, batch_patch_rays_d, batch_patch_gt_depth, batch_patch_gt_color):

    last_pose = last_pose.clone().float()
    last_pose[:,:3,1:3] = last_pose[:,:3,1:3] * (-1)
    init_pose = init_pose.clone().unsqueeze(0)
    init_pose[:,:3,1:3] = init_pose[:,:3,1:3] * (-1)

    batch_patch_rays_o = batch_patch_rays_o.unsqueeze(0).cuda()
    batch_patch_rays_d = batch_patch_rays_d.unsqueeze(0).cuda()
    
    patch_size = 1
    fx, fy, cx, cy = intrinsics[0].item(), intrinsics[1].item(), intrinsics[2].item(), intrinsics[3].item()

    channel = images.shape[-1]
    batch_patch_gt_depth = batch_patch_gt_depth.view( patch_size *  patch_size, -1 )
    batch_patch_gt_color = batch_patch_gt_color.view( patch_size *  patch_size, -1, channel)

    patch_3d_pts = (batch_patch_rays_o + batch_patch_rays_d * batch_patch_gt_depth[:, :, None]).float()
    uv, z = project_point3d_to_image_batch(
            last_pose, patch_3d_pts.view(-1, 3, 1), fx, fy, cx, cy, patch_3d_pts.device)
    edge = 0
    uv = uv.view(patch_3d_pts.shape[0], patch_3d_pts.shape[1], last_pose.shape[0], 2)
    mask = ((uv[(patch_size * patch_size) // 2, :, :, 0] < W - edge)
            * (uv[(patch_size * patch_size) // 2, :, :, 0] > edge)
            * (uv[(patch_size * patch_size) // 2, :, :, 1] < H - edge)
            * (uv[(patch_size * patch_size) // 2, :, :, 1] > edge))
    mask = mask & (z.view(patch_3d_pts.shape[0], patch_3d_pts.shape[1], last_pose.shape[0], 1
            )[(patch_size * patch_size) // 2, :, :, 0]<= 0)

    windows_reproj_idx = uv.permute(2, 1, 0, 3)
    windows_reproj_idx[..., 0] = windows_reproj_idx[..., 0] / W * 2.0 - 1.0
    windows_reproj_idx[..., 1] = windows_reproj_idx[..., 1] / H * 2.0 - 1.0

    windows_reproj_gt_color = torch.nn.functional.grid_sample(last_image.permute(0, 3, 1, 2).float(),
        windows_reproj_idx, padding_mode="border",).permute( 2, 0, 3, 1 )
    windows_reproj_gt_color = windows_reproj_gt_color.squeeze(2)
    windows_reproj_gt_color = windows_reproj_gt_color.squeeze(2)
    batch_gt_color = batch_patch_gt_color.permute(1, 0, 2).squeeze(1)

    color_residual = torch.abs(windows_reproj_gt_color - batch_gt_color)
    warp_mask = ((color_residual < 0.6) & (windows_reproj_gt_color > 0)) | ((windows_reproj_gt_color < batch_gt_color)&(windows_reproj_gt_color > 0))

    finial_mask = warp_mask.all(dim=1)
    return finial_mask