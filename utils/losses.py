import torch
# This now imports the 'emd' module, which is the correct name for the compiled library.
import emd
# This now uses the correct path and function names from your project's 'external' folder.
from external.ChamferDistancePytorch.chamfer_python import distChamfer, distChamfer_raw

#SANKET'S OWN LOSSES.PY FILE TO SELECTIVELY CHOOSE NECESSARY LOSS AND THEN GET THE COMPLETION RESULT

def calc_cd(pcd1, pcd2):
    """
    Calculates the Chamfer Distance using the project's existing implementation.
    """
    dist1, dist2, _, _ = distChamfer(pcd1, pcd2)
    # Return the mean of the batch loss
    cd_loss = (dist1.mean(1) + dist2.mean(1)).mean()
    return cd_loss

def calc_emd(pcd1, pcd2, eps=0.005, iterations=50):
    """
    Calculates the Earth Mover's Distance.
    """
    emd_loss_func = emd.emdModule()
    dist, _ = emd_loss_func(pcd1, pcd2, eps, iterations)
    # Return the mean of the batch loss
    emd_loss = torch.sqrt(dist).mean()
    return emd_loss

def calc_dcd(x, gt, alpha=1000, n_lambda=1):
    """
    Calculates the Density-aware Chamfer Distance.
    """
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    _, n_gt, _ = gt.shape

    assert x.shape == gt.shape, "DCD requires point clouds to have the same number of points."

    # Use the raw Chamfer distance to get nearest neighbors and their indices
    dist1, dist2, idx1, idx2 = distChamfer_raw(gt, x)

    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    loss1 =[]
    loss2 =[]
    for b in range(batch_size):
        count1 = torch.bincount(idx1[b], minlength=n_x)
        weight1 = count1[idx1[b].long()].float().detach() ** n_lambda
        weight1 = (weight1 + 1e-6) ** (-1)
        loss1.append((- exp_dist1[b] * weight1 + 1.).mean())

        count2 = torch.bincount(idx2[b], minlength=n_gt)
        weight2 = count2[idx2[b].long()].float().detach() ** n_lambda
        weight2 = (weight2 + 1e-6) ** (-1)
        loss2.append((- exp_dist2[b] * weight2 + 1.).mean())

    loss1 = torch.stack(loss1)
    loss2 = torch.stack(loss2)
    loss = (loss1 + loss2) / 2

    # Return the mean of the batch loss
    return loss.mean()