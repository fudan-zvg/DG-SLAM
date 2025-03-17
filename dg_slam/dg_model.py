import torch
import numpy as np
from lietorch import SE3
from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from gs_tracking_mapping import gs_tracking_mapping
from dg_slam.pose_transform import quaternion_to_transform_noBatch

def pose_matrix_from_quaternion(pvec):
    """ convert 4x4 pose matrix to (t, q) """
    from scipy.spatial.transform import Rotation

    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
    pose[:3, 3] = pvec[:3]
    return pose

class dg_model:
    def __init__(self, cfg, args):
        super(dg_model, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis
        self.cfg = cfg

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)
        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)
        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        self.tracking_mapping = gs_tracking_mapping(self.cfg, self.args, self.video)
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

        self.mapping_counter = 0
        self.inv_pose = None

    def load_weights(self, weights):
        """ load trained model weights """
        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth, pose, intrinsics, seg_mask):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, pose, intrinsics, seg_mask)
            # local bundle adjustment
            update_status = self.frontend()

        if update_status or (tstamp == self.cfg["data"]["n_img"] - 1):
            tracking_min = self.frontend.graph.ii.min().item()
            tracking_max = self.frontend.graph.ii.max().item()
            while(True):
                if (self.mapping_counter < tracking_min) or ((self.mapping_counter < tracking_max + 1) and (tstamp == self.cfg["data"]["n_img"] - 1)):
                    idx = self.mapping_counter
                    idx = torch.tensor(idx)
                    img_idx = self.video.images[self.mapping_counter] / 255
                    
                    disps_up = self.video.disps_up[self.mapping_counter]
                    depth_idx = torch.where(disps_up > 0, 1.0/disps_up, disps_up)

                    gt_pose_idx = self.video.poses_gt[self.mapping_counter]
                    gt_pose_idx = torch.from_numpy(pose_matrix_from_quaternion(gt_pose_idx.cpu())).cuda() 

                    pose_idx = self.video.poses[self.mapping_counter] 
                    pose_idx = quaternion_to_transform_noBatch(SE3(pose_idx).inv().data) 

                    if self.inv_pose is None:
                        init_pose = gt_pose_idx
                        self.inv_pose = torch.inverse(init_pose)
                        gt_pose_idx = self.inv_pose @ gt_pose_idx
                    else:
                        gt_pose_idx = self.inv_pose @ gt_pose_idx

                    seg_mask_idx = ~ self.video.seg_masks_ori[self.mapping_counter]

                    self.tracking_mapping.run(idx, img_idx, depth_idx, gt_pose_idx, pose_idx, seg_mask_idx)
                    self.mapping_counter += 1
                else:
                    break

    def terminate_woBA(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """
        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy() # c2w