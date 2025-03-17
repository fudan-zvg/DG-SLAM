import os
import os.path as osp
import pickle
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import yaml
import torch.nn.functional as F

class BasicLogger:
    def __init__(self) -> None:
        self.img_dir = None

    def get_random_time_str(self):
        return datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")

    def log_ckpt(self, mapper):
        decoder_state = {f: v.cpu()
                         for f, v in mapper.decoder.state_dict().items()}
        map_state = {f: v.cpu() for f, v in mapper.map_states.items()}
        embeddings = mapper.embeddings.cpu()
        svo = mapper.svo
        torch.save({
            "decoder_state": decoder_state,
            "map_state": map_state,
            "embeddings": embeddings,
            "svo": svo},
            os.path.join(self.ckpt_dir, "final_ckpt.pth"))

    def log_config(self, config):
        out_path = osp.join(self.backup_dir, "config.yaml")
        yaml.dump(config, open(out_path, 'w'))

    def log_mesh(self, mesh, name="final_mesh.ply"):
        out_path = osp.join(self.mesh_dir, name)
        o3d.io.write_triangle_mesh(out_path, mesh)

    def log_point_cloud(self, pcd, name="final_points.ply"):
        out_path = osp.join(self.mesh_dir, name)
        o3d.io.write_point_cloud(out_path, pcd)

    def log_numpy_data(self, data, name, ind=None):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if ind is not None:
            np.save(osp.join(self.misc_dir, "{}-{:05d}.npy".format(name, ind)), data)
        else:
            np.save(osp.join(self.misc_dir, f"{name}.npy"), data)

    def log_debug_data(self, data, idx):
        with open(os.path.join(self.misc_dir, f"scene_data_{idx}.pkl"), 'wb') as f:
            pickle.dump(data, f)

    def log_raw_image(self, ind, rgb, depth):
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        rgb = cv2.cvtColor(rgb*255, cv2.COLOR_RGB2BGR)
        cv2.imwrite(osp.join(self.img_dir, "{:05d}.jpg".format(
            ind)), (rgb).astype(np.uint8))
        cv2.imwrite(osp.join(self.img_dir, "{:05d}.png".format(
            ind)), (depth*5000).astype(np.uint16))