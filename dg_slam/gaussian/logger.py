import os
import torch

class Logger(object):
    """
    Save checkpoints to file.

    """

    def __init__(self, cfg, args, mapper
                 ):
        self.verbose = mapper.verbose
        self.ckptsdir = mapper.ckptsdir
        self.gt_c2w_list = mapper.gt_c2w_list
        self.estimate_c2w_list = mapper.estimate_c2w_list

    def log(self, idx, keyframe_dict, keyframe_list, selected_keyframes, gaussians, exposure_feat=None):
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx))
        torch.save({
            'xyz': gaussians.get_xyz().detach().cpu().numpy(),
            'features_dc': gaussians.get_features_dc().detach().cpu().numpy(),
            'features_rest': gaussians.get_features_rest().detach().cpu().numpy(),
            'scaling': gaussians.get_scaling().detach().cpu().numpy(),
            'rotation': gaussians.get_rotation().detach().cpu().numpy(),
            'opacity': gaussians.get_opacity().detach().cpu().numpy(),
            'pts_num': gaussians.pts_num(),
            'input_pos': gaussians.input_pos(),
            'input_rgb': gaussians.input_rgb(),
            'gt_c2w_list': self.gt_c2w_list,
            'estimate_c2w_list': self.estimate_c2w_list,
            'keyframe_list': keyframe_list,
            'keyframe_dict': keyframe_dict,
            'selected_keyframes': selected_keyframes,
            'idx': idx,
            "exposure_feat_all": torch.stack(exposure_feat, dim=0)
            if exposure_feat is not None
            else None,
        }, path)

        if self.verbose:
            print('Saved checkpoints at', path)
