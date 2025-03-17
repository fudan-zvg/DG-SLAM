import numpy as np
import os.path as osp

import torch
from lietorch import SE3

import geom.projective_ops as pops
from scipy.spatial.transform import Rotation

def all_pairs_distance_matrix(poses, beta=2.5):
    """ compute distance matrix between all pairs of poses """
    poses = np.array(poses, dtype=np.float32)
    poses[:,:3] *= beta # scale to balence rot + trans
    poses = SE3(torch.from_numpy(poses))

    r = (poses[:,None].inv() * poses[None,:]).log()
    return r.norm(dim=-1).cpu().numpy()

def pose_matrix_to_quaternion(pose):
    """ convert 4x4 pose matrix to (t, q) """
    q = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return np.concatenate([pose[:3, 3], q], axis=0)

def compute_distance_matrix_flow(poses, disps, intrinsics):
    """ compute flow magnitude between all pairs of frames """
    if not isinstance(poses, SE3):
        poses = torch.from_numpy(poses).float().cuda()[None]
        poses = SE3(poses).inv()

        disps = torch.from_numpy(disps).float().cuda()[None]
        intrinsics = torch.from_numpy(intrinsics).float().cuda()[None]

    N = poses.shape[1]
    
    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
    ii = ii.reshape(-1).cuda()
    jj = jj.reshape(-1).cuda()

    MAX_FLOW = 100.0
    matrix = np.zeros((N, N), dtype=np.float32)

    s = 2048
    for i in range(0, ii.shape[0], s):
        flow1, val1 = pops.induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s])
        flow2, val2 = pops.induced_flow(poses, disps, intrinsics, jj[i:i+s], ii[i:i+s])
        
        flow = torch.stack([flow1, flow2], dim=2)
        val = torch.stack([val1, val2], dim=2)
        
        mag = flow.norm(dim=-1).clamp(max=MAX_FLOW)
        mag = mag.view(mag.shape[1], -1)
        val = val.view(val.shape[1], -1)

        mag = (mag * val).mean(-1) / val.mean(-1)
        mag[val.mean(-1) < 0.7] = np.inf

        i1 = ii[i:i+s].cpu().numpy()
        j1 = jj[i:i+s].cpu().numpy()
        matrix[i1, j1] = mag.cpu().numpy()

    return matrix

def compute_distance_matrix_flow2(poses, disps, intrinsics, beta=0.4):
    """ compute flow magnitude between all pairs of frames """
    N = poses.shape[1]
    
    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)

    MAX_FLOW = 128.0
    matrix = np.zeros((N, N), dtype=np.float32)

    s = 2048
    for i in range(0, ii.shape[0], s):
        flow1a, val1a = pops.induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s], tonly=True)
        flow1b, val1b = pops.induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s])
        flow2a, val2a = pops.induced_flow(poses, disps, intrinsics, jj[i:i+s], ii[i:i+s], tonly=True)
        flow2b, val2b = pops.induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s])

        flow1 = flow1a + beta * flow1b
        val1 = val1a * val2b

        flow2 = flow2a + beta * flow2b
        val2 = val2a * val2b
        
        flow = torch.stack([flow1, flow2], dim=2)
        val = torch.stack([val1, val2], dim=2)
        
        mag = flow.norm(dim=-1).clamp(max=MAX_FLOW)
        mag = mag.view(mag.shape[1], -1)
        val = val.view(val.shape[1], -1)

        mag = (mag * val).mean(-1) / val.mean(-1)
        mag[val.mean(-1) < 0.8] = np.inf

        i1 = ii[i:i+s].cpu().numpy()
        j1 = jj[i:i+s].cpu().numpy()
        matrix[i1, j1] = mag.cpu().numpy()

    return matrix
