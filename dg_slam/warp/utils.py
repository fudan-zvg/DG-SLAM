import torch
import numpy as np

def get_sample_uv_by_indices_batch(H0, H1, W0, W1, depth, color, indices, idxs, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    if depth is not None:
        depth = depth[:, H0:H1, W0:W1]
    else:
        pass
    color = color[:, H0:H1, W0:W1]

    # compute new idxs
    channel = color.shape[-1]
    
    indices=indices
    if depth is not None:
        depth = torch.nn.functional.grid_sample(depth.view(list(depth.shape[0:])+[1]).permute(0, 3, 1, 2),
                            indices.view(indices.shape[0],depth.shape[0],-1,2).permute(1,0,2,3),
                            mode="nearest").permute(2,0,3,1).reshape(-1)
    color = torch.nn.functional.grid_sample(color.permute(0, 3, 1, 2),
                          indices.view(indices.shape[0],color.shape[0],-1,2).permute(1,0,2,3),
                          mode="bilinear").permute(2,0,3,1).contiguous().view(-1,channel)
    
    # color = color[]    
    i = indices[...,0].view(-1)
    j = indices[...,1].view(-1)
    return i, j, depth, color

def get_rays_from_uv_batch(i, j, c2w, H, W, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)

    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(i.shape[0],i.shape[1], 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[None, :, :3, :3], -1)
    rays_o = c2w[:, :3, -1].expand(rays_d.shape)
    return rays_o, rays_d
    
def get_samples_by_indices_batch(H0,
                           H1,
                           W0,
                           W1,
                           H,
                           W,
                           fx,
                           fy,
                           cx,
                           cy,
                           c2w,
                           depth,
                           color,
                           indices,
                           device="cuda:0",
                           return_uv=False):
    """get n rays from the image region 
    c2w is the camera pose and depth/color is the corresponding image tensor.
    select the rays by the given indices

    Args:
        H0 (_type_): _description_
        H1 (_type_): _description_
        W0 (_type_): _description_
        W1 (_type_): _description_
        n (_type_): _description_
        H (_type_): _description_
        W (_type_): _description_
        fx (_type_): _description_
        fy (_type_): _description_
        cx (_type_): _description_
        cy (_type_): _description_
        c2w (_type_): _description_
        depth (_type_): _description_
        color (_type_): _description_
        indices (_type_): [-1,1] , WH , for grid sample
        device (_type_): _description_
    """

    i, j, sample_depth, sample_color = get_sample_uv_by_indices_batch(
        H0, H1, W0, W1, depth, color, indices, device)
    i = (indices[..., 0].view(-1) + 1) / 2.0 * W
    j = (indices[..., 1].view(-1) + 1) / 2.0 * H
    i = i.view(-1,c2w.shape[0])
    j = j.view(-1,c2w.shape[0])
    rays_o, rays_d = get_rays_from_uv_batch(i, j, c2w, H, W, fx, fy, cx, cy, device)
    if not return_uv:
        return rays_o, rays_d, sample_depth, sample_color
    else:
        return rays_o, rays_d, sample_depth, sample_color, torch.stack([i,j],dim=1).to(device)

def project_point3d_to_image_batch(c2ws, pts3d, fx,fy, cx, cy, device="cuda:0"):
    if pts3d.shape[-2] == 3:
        pts3d_homo = torch.cat([pts3d, torch.ones_like(pts3d[:,0].view(-1,1,1))], dim=-2)
    elif pts3d.shape[-2] == 4:
        pts3d_homo = pts3d
    else:
        raise NotImplementedError
    
    pts3d_homo = pts3d_homo.to(device)
    w2cs = torch.inverse(c2ws)
    
    pts2d_homo = w2cs @ pts3d_homo[:,None,:,:] # [Cn, 4, 4] @ [Pn, 1, 4, 1] = [Pn, Cn, 4, 1]
    pts2d = pts2d_homo[:,:,:3]
    K = torch.from_numpy(
        np.array([[fx, .0, cx], [.0, fy, cy],
                  [.0, .0, 1.0]]).reshape(3, 3)).to(device).float()
    pts2d[:,:,0] *= -1
    uv = K @ pts2d 
    z = uv[:,:,-1:] + 1e-5
    uv = uv[:,:,:2]/z  
    
    uv = uv.float()
    return uv,z