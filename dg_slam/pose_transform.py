import torch
import torch.nn.functional as F

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(*batch_dim, 4)

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def quaternion_to_transform_noBatch(quaternions):
    #  Q: i,j,k,w
    quaternion_Q = quaternions[3:][[3,0,1,2]]
    quaternion_T = quaternions[0:3]
    transform = torch.eye(4).to(quaternions.device)
    transform[:3,:3] = quaternion_to_matrix(quaternion_Q)
    transform[:3, 3] = quaternion_T
    return transform


def quaternion_to_transform(quaternions):
    b = quaternions.shape[0]
    quaternion_Q = quaternions[:,3:]
    quaternion_T = quaternions[:,0:3]
    transform = torch.zeros(b,3,4).to(quaternions.device)
    transform[:,:3,:3] = quaternion_to_matrix(quaternion_Q)
    transform[:,:3, 3] = quaternion_T
    return transform

def transform_to_quaternion(transform: torch.Tensor) -> torch.Tensor:
    transform_Q = transform[:3,:3]
    transform_T = transform[:3, 3]
    quaternions = torch.zeros(7).to(transform.device)
    quaternions[3:] = matrix_to_quaternion(transform_Q)
    quaternions[3:] = quaternions[3:][[1,2,3,0]]
    quaternions[0:3] = transform_T
    return quaternions

def intrinsicMatrix_to_list(matrix):
    batch = matrix.shape[0]
    frames = matrix.shape[1]
    intrinsic = torch.zeros(batch, frames ,4).to(dtype = torch.float32, device= matrix.device)
    intrinsic[:,:,0] = matrix[:,:,0,0]
    intrinsic[:,:,1] = matrix[:,:,1,1]
    intrinsic[:,:,2] = matrix[:,:,0,2]
    intrinsic[:,:,3] = matrix[:,:,1,2]
    return intrinsic