import torch
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment

# --- Local Project Imports ---
# These are the only functions we need from your local 'ChamferDistancePytorch' library
from external.ChamferDistancePytorch.chamfer_python import distChamfer, distChamfer_raw


# =======================================================================
# 1. HYPERBOLIC HELPER FUNCTION (For HyperCD)
# =======================================================================

def arcosh(x, eps=1e-5):
    """
    Computes the inverse hyperbolic cosine (arcosh), used for distance in hyperbolic space.
    """
    # Clamps x to ensure it's >= 1 + eps for stable log and sqrt calculation
    x = torch.clamp(x, min=1 + eps)
    return torch.log(x + torch.sqrt(1 + x) * torch.sqrt(x - 1))


# =======================================================================
# 2. BASELINE LOSSES (CD, EMD, DCD)
# =======================================================================

def calc_cd(pcd1, pcd2):
    """Calculates the standard Chamfer Distance (L2^2) using k=1 NN."""
    dist1, dist2, _, _ = distChamfer(pcd1, pcd2)
    cd_loss = (dist1.mean(1) + dist2.mean(1)).mean()
    return cd_loss


def calc_emd(x, gt, eps=0.005, iterations=300):
    """
    Compute Earth Mover's Distance (EMD) loss.
    """
    x = Variable(x)
    gt = Variable(gt)

    try:
        # --- CUDA-based EMD module ---
        from external.emd.emd_module import emdModule
        emd = emdModule()
        dist, _ = emd(x, gt, eps, iterations)
        return torch.sqrt(dist).mean()

    except Exception as e:
        # CPU fallback (kept for robustness)
        print(f"[WARN] Falling back to CPU-based EMD (slow). Reason: {e}")
        x_np = x.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()

        total_loss = 0.0
        for i in range(len(x_np)):
            dist_matrix = np.linalg.norm(x_np[i][:, None, :] - gt_np[i][None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            total_loss += dist_matrix[row_ind, col_ind].mean()

        return torch.tensor(total_loss / len(x_np), dtype=torch.float32, device=x.device)


def calc_dcd(x, gt, alpha=1000, n_lambda=1):
    """
    Calculates the Density-aware Chamfer Distance (DCD).
    """
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    _, n_gt, _ = gt.shape

    assert x.shape == gt.shape, "DCD requires point clouds to have the same number of points."

    # Use distChamfer to get k=1 indices and distances.
    chamfer_out = distChamfer(gt, x)

    if isinstance(chamfer_out, tuple) and len(chamfer_out) == 4:
        dist1, dist2, idx1, idx2 = chamfer_out
    else:
        # Fallback if distChamfer only returns one tuple
        dist1 = chamfer_out
        dist2 = chamfer_out
        idx1 = torch.zeros(batch_size, n_gt, dtype=torch.long, device=x.device)
        idx2 = torch.zeros(batch_size, n_x, dtype=torch.long, device=x.device)

    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    loss1 = []
    loss2 = []
    for b in range(batch_size):
        # Density weighting using torch.bincount (DCD core logic)
        count1 = torch.bincount(idx1[b], minlength=n_x)
        weight1 = count1[idx1[b].long()].float().detach() ** n_lambda
        weight1 = (weight1 + 1e-6) ** (-1)
        loss1.append((-exp_dist1[b] * weight1 + 1.).mean())

        count2 = torch.bincount(idx2[b], minlength=n_gt)
        weight2 = count2[idx2[b].long()].float().detach() ** n_lambda
        weight2 = (weight2 + 1e-6) ** (-1)
        loss2.append((-exp_dist2[b] * weight2 + 1.).mean())

    loss1 = torch.stack(loss1)
    loss2 = torch.stack(loss2)
    loss = (loss1 + loss2) / 2

    return loss.mean()


# =======================================================================
# 3. INFOCD LOSSES (Contrastive/Entropy-based Loss)
# =======================================================================

def calc_infocd(p1, p2):
    """
    Computes InfoCD Loss (Symmetric).
    Based on log-likelihood of nearest neighbor assignments (InfoCD core logic).
    """
    # Use distChamfer to get the standard 4-tuple outputs
    dist1_raw, dist2_raw, _, _ = distChamfer(p1, p2)
    
    # Square root distances for L1 norm analogy
    d1 = torch.clamp(dist1_raw, min=1e-9).sqrt()
    d2 = torch.clamp(dist2_raw, min=1e-9).sqrt()

    # The InfoCD formula: -log(exp(-d) / sum(exp(-d)))
    distances1 = -torch.log(
        torch.exp(-0.5 * d1) / 
        (torch.sum(torch.exp(-0.5 * d1) + 1e-7, dim=-1).unsqueeze(-1))**1e-7
    )
                             
    distances2 = -torch.log(
        torch.exp(-0.5 * d2) / 
        (torch.sum(torch.exp(-0.5 * d2) + 1e-7, dim=-1).unsqueeze(-1))**1e-7
    )

    return (torch.sum(distances1) + torch.sum(distances2)) / 2


def calc_infocd_one_side(p1, p2):
    """
    Computes InfoCD Loss (Single-Sided). Used for partial matching loss.
    """
    dist1_raw, dist2_raw, idx1, idx2 = distChamfer(p1, p2)
    dist1 = torch.clamp(dist1_raw, min=1e-9)
    d1 = torch.sqrt(dist1)

    # InfoCD formula (single side)
    distances1 = -torch.log(
        torch.exp(-0.5 * d1) / 
        (torch.sum(torch.exp(-0.5 * d1) + 1e-7, dim=-1).unsqueeze(-1))**1e-7
    )

    return torch.sum(distances1)


# =======================================================================
# 4. HYPERCD LOSSES (Hyperbolic Geometry Loss)
# =======================================================================

def calc_hypercd(p1, p2):
    """
    Computes HyperCD Loss (Symmetric). Distances are transformed into hyperbolic space.
    """
    # Get standard Chamfer distances (L2^2)
    dist1_raw, dist2_raw, _, _ = distChamfer(p1, p2)
    
    # Convert Euclidean Chamfer Distance to Hyperbolic Arcosh distance
    # The term '1 * d1' corresponds to the parameter kappa=1 in the Poincare Ball model.
    d1 = arcosh(1 + 1 * dist1_raw)
    d2 = arcosh(1 + 1 * dist2_raw)
    
    # Return mean total distance
    return torch.mean(d1) + torch.mean(d2)


def calc_hypercd_one_side(p1, p2):
    """
    Computes HyperCD Loss (Single-Sided).
    """
    dist1_raw, dist2_raw, _, _ = distChamfer(p1, p2)
    
    # Convert Euclidean Chamfer Distance to Hyperbolic Arcosh distance
    d1 = arcosh(1 + 1 * dist1_raw)
    
    return torch.mean(d1)


# =======================================================================
# 5. UNIFORMCD LOSS (Fixed to work with project libraries)
# =======================================================================

def calc_uniform_chamfer_loss_tensor(x, y, k=32, return_assignment=False, return_dists=False):
    """
    Calculates the Uniform Chamfer Distance (RE-WRITTEN to use distChamfer_raw).
    This version is compatible with your existing project libraries.
    """
    eps = 0.00001
    k2 = 32  # k for density check
    power = 2

    # --- Find k-Nearest Neighbors using distChamfer_raw and torch.topk ---
    
    # 1. Find k-NN for (x -> y) and (y -> x)
    dist_xy_full = distChamfer_raw(x, y) # [B, N_x, N_y]
    dist_xy, idx_xy = torch.topk(dist_xy_full, k=k, dim=2, largest=False)
    
    dist_yx_full = distChamfer_raw(y, x) # [B, N_y, N_x]
    dist_yx, idx_yx = torch.topk(dist_yx_full, k=k, dim=2, largest=False)
    
    # 2. Find k-NN for (x -> x) and (y -> y) for density (Avoiding distance to self)
    dist_xx_full = distChamfer_raw(x, x) + torch.eye(x.shape[1], device=x.device) * 1e9 # [B, N_x, N_x]
    dist_xx, _ = torch.topk(dist_xx_full, k=k2, dim=2, largest=False)
    
    dist_yy_full = distChamfer_raw(y, y) + torch.eye(y.shape[1], device=y.device) * 1e9 # [B, N_y, N_y]
    dist_yy, _ = torch.topk(dist_yy_full, k=k2, dim=2, largest=False)
    
    # --- The original uniformCD logic ---
    
    # --- Forward Direction (x -> y) ---
    nn_x_dists = dist_xx
    nn_xy_dists = dist_xy
    nn_xy_idx = idx_xy
    nn_yx_dists = dist_yx
    nn_yx_idx = idx_yx  

    # 1. measure density of x with itself
    density_x = torch.mean(nn_x_dists, dim=2) 
    density_x = 1 / (density_x + eps) 
    high, low = torch.max(density_x), torch.min(density_x)
    diff = high - low
    density_x = (density_x - low) / (diff + eps) # Normalize
    
    # 2. measure density of x with other cloud y
    density_xy = torch.mean(nn_xy_dists, dim=2)
    density_xy = 1 / (density_xy + eps)
    high, low = torch.max(density_xy), torch.min(density_xy)
    diff = high - low
    density_xy = (density_xy - low) / (diff + eps) # Normalize
    
    # 3. Calculate weights
    w_x = torch.div(density_xy, density_x + eps) 
    w_x = torch.pow(w_x, power)
    scaling_factors_1 = w_x.unsqueeze(2).repeat(1, 1, k)
    
    multiplier = torch.gather(scaling_factors_1, 1, nn_yx_idx)
    
    # 4. Apply weights
    scaled_dist_1 = torch.mul(nn_yx_dists, multiplier)
    scaled_dist_1x, i1 = torch.min(scaled_dist_1, 2)
        
    
    # --- Backward Direction (y -> x) ---
    nn_y_dists = dist_yy
    nn_yx_dists = dist_yx
    nn_yx_idx = idx_yx
    nn_xy_dists = dist_xy
    nn_xy_idx = idx_xy 

    # 1. measure density of y with itself
    density_y = torch.mean(nn_y_dists, dim=2)
    density_y = 1 / (density_y + eps)
    high, low = torch.max(density_y), torch.min(density_y)
    diff = high - low
    density_y = (density_y - low) / (diff + eps) # Normalize
    
    # 2. measure density of y with other cloud x
    density_yx = torch.mean(nn_yx_dists, dim=2)
    density_yx = 1 / (density_yx + eps)
    high, low = torch.max(density_yx), torch.min(density_yx)
    diff = high - low
    density_yx = (density_yx - low) / (diff + eps) # Normalize

    # 3. Calculate weights
    w_y = torch.div(density_yx, density_y + eps) # Add eps for stability
    w_y = torch.pow(w_y, power)
    scaling_factors_0 = w_y.unsqueeze(2).repeat(1, 1, k)
    multiplier = torch.gather(scaling_factors_0, 1, nn_xy_idx)
    
    # 4. Apply weights
    scaled_dist_0 = torch.mul(nn_xy_dists, multiplier)
    scaled_dist_0x, i0 = torch.min(scaled_dist_0, 2)
    
    # --- Final Calculation ---
    min_dist_1 = torch.gather(nn_yx_dists, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0]
    min_dist_0 = torch.gather(nn_xy_dists, 2, i0.unsqueeze(2).repeat(1,1,k))[:, :, 0]
    
    uniform_cd = torch.sum(torch.sqrt(min_dist_1)) + torch.sum(torch.sqrt(min_dist_0))
    
    batch_size, point_count, _ = x.shape
    
    bidirectional_dist = uniform_cd
    bidirectional_dist = bidirectional_dist / (batch_size) # Average over batch
    
    if return_dists:
        return min_dist_0, min_dist_1
    
    if return_assignment:
        min_ind_1 = torch.gather(nn_yx_idx, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0]
        min_ind_0 = torch.gather(nn_xy_idx, 2, i0.unsqueeze(2).repeat(1,1,k))[:, :, 0]
        
        return bidirectional_dist, [min_ind_0.detach().cpu().numpy(), min_ind_1.detach().cpu().numpy()]
    else:
        return bidirectional_dist