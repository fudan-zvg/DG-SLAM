## Differential Gaussian Rasterization for DG-SLAM

This is a modified Differential 3DGS(3D Gaussian Splatting) Rasterization version based on [orgin 3DGS Rasterization implementation](https://github.com/graphdeco-inria/diff-gaussian-rasterization/tree/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d).

We add some addition support for the DG-SLAM:

- Depth forward rendering 
- Depth rendering backward
- Pose optimization backward

### ⚙️ Installation

Please follow the instructions below to install the dependencies.
```bash
python setup.py install
pip install .
```

**NOTE**: The  input parameters and rasterization operations have diverged significantly from the original 3DGS codebase to align with specific requirements and match the desigend structure of DG-SLAM.

### 📜 BibTeX

If you find this project helpful, please consider citing the following our paper and original Differential Gaussian Rasterization:

```
@inproceedings{xu2024dgslam,
title={{DG}-{SLAM}: Robust Dynamic Gaussian Splatting {SLAM} with Hybrid Pose Optimization},
author={Yueming Xu and Haochen Jiang and Zhongyang Xiao and Jianfeng Feng and Li Zhang},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=tGozvLTDY3}
}
```

```
@article{kerbl20233d,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering.},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal={ACM Trans. Graph.},
  volume={42},
  number={4},
  pages={139--1},
  year={2023}
}
```