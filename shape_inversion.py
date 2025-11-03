import os
import os.path as osp
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from model.treegan_network import Generator, Discriminator

from utils.common_utils import *
from loss import *
from evaluation.pointnet import *
import time
from external.ChamferDistancePytorch.chamfer_python import distChamfer, distChamfer_raw

import random

# ---------------------------------------
# Device setup: use GPU if available else CPU
# ---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Optional: make runs reproducible
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)


class ShapeInversion(object):

    def __init__(self, args):
        self.args = args

        if self.args.dist:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank, self.world_size = 0, 1

        # init seed for static masks: ball_hole, knn_hole, voxel_mask
        self.to_reset_mask = True
        self.mask_type = self.args.mask_type
        self.update_G_stages = self.args.update_G_stages
        self.iterations = self.args.iterations
        self.G_lrs = self.args.G_lrs
        self.z_lrs = self.args.z_lrs
        self.select_num = self.args.select_num
        self.loss_log = []
        
        self.device = getattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # create model
        self.G = Generator(features=args.G_FEAT, degrees=args.DEGREE, support=args.support, args=self.args).cuda()
        self.D = Discriminator(features=args.D_FEAT).to(device)
        # Use provided device if present, else auto-detect
        # self.device = getattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # # === Instantiate Generator exactly as checkpoint expects ===
        # # This matches the keys you showed from chair.pt
        # self.G = Generator(
        #     features=[96, 256, 256, 256, 128, 128, 128, 3],
        #     degrees=[1, 1, 2, 4, 8, 16, 32],
        #     support=args.support,
        #     args=self.args
        # ).to(self.device)

        # # Instantiate Discriminator using args (keep as before)
        # self.D = Discriminator(features=args.D_FEAT).to(self.device)

        # # === Load checkpoint safely (map to correct device) ===
        # ckpt_path = getattr(args, "checkpoint", None) or getattr(args, "ckpt_load", None)
        # if ckpt_path is None or not os.path.exists(ckpt_path):
        #     raise FileNotFoundError(f"Checkpoint not found. Set args.checkpoint or args.ckpt_load to valid path (tried: {ckpt_path})")

        # checkpoint = torch.load(ckpt_path, map_location=self.device)

        # # Load Generator weights (strict=True ensures exact match)
        # try:
        #     self.G.load_state_dict(checkpoint['G_state_dict'], strict=True)
        #     print(f"[INFO] Generator weights loaded (strict=True) from {ckpt_path}")
        # except RuntimeError as e:
        #     # If strict fails, raise with informative message (so you can inspect remaining mismatches)
        #     raise RuntimeError(f"Generator state_dict strict load failed. Check model config to match checkpoint. Original error:\n{e}")

        # # Load Discriminator weights (also strict)
        # if 'D_state_dict' in checkpoint:
        #     try:
        #         self.D.load_state_dict(checkpoint['D_state_dict'], strict=True)
        #         print(f"[INFO] Discriminator weights loaded (strict=True) from {ckpt_path}")
        #     except RuntimeError as e:
        #         raise RuntimeError(f"Discriminator state_dict strict load failed. Original error:\n{e}")
        # else:
        #     print("[WARN] 'D_state_dict' not found in checkpoint")

        # Store generator weights on CPU to avoid GPU OOM when deepcopying / restoring
        self.G_weight = {k: v.cpu().clone() for k, v in self.G.state_dict().items()}
        # free any cached GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        print("Current Generator architecture:")
        print(f"Number of GCN layers: {self.G.layer_num}")
        print(f"Degrees: {self.args.DEGREE}")
        print(f"Features: {self.args.G_FEAT}")

        # Print each TreeGCN layer parameters properly
        for i, layer in enumerate(self.G.gcn):
            print(f"\nLayer {i} parameters:")
            for name, param in layer.named_parameters():
                print(f"  {name}: {param.shape}")


        self.G.optim = torch.optim.Adam(
            [{'params': self.G.get_params(i)}
                for i in range(self.G.layer_num)],
            lr=self.G_lrs[0],
            betas=(float(0), float(0.99)),  # Explicitly cast betas to floats
            weight_decay=0,
            eps=1e-8)
        self.z = torch.zeros((1, 1, 96)).normal_().cuda()
        self.z = Variable(self.z, requires_grad=True)
        self.z_optim = torch.optim.Adam([self.z], lr=self.args.z_lrs[0], betas=(float(0), float(0.99)))  # Explicitly cast betas to floats

        # load weights
        checkpoint = torch.load(args.ckpt_load, map_location=self.args.device)

        # ---- SAFE LOAD CHECKPOINT FOR G ----
        if 'G_state_dict' in checkpoint:
            ckpt_g = checkpoint['G_state_dict']
            model_g = self.G.state_dict()

            matched = {}
            skipped = []
            for k, v in ckpt_g.items():
                if k in model_g and v.shape == model_g[k].shape:
                    matched[k] = v
                else:
                    skipped.append((k, tuple(v.shape), model_g.get(k, None).shape if k in model_g else 'not_in_model'))

            self.G.load_state_dict(matched, strict=False)
            print(f"[INFO] Loaded {len(matched)} matching keys into Generator; skipped {len(skipped)} mismatched layers.")
            if skipped:
                print("Skipped (key, ckpt_shape, model_shape):")
                for s in skipped[:20]:
                    print(" ", s)
        else:
            print("[WARN] No 'G_state_dict' found in checkpoint.")

        # ---- SAFE LOAD CHECKPOINT FOR D (try strict then fallback) ----
        if 'D_state_dict' in checkpoint:
            try:
                self.D.load_state_dict(checkpoint['D_state_dict'])
                print("[INFO] Discriminator: loaded checkpoint with strict=True")
            except RuntimeError as e:
                print("[WARN] Discriminator strict load failed:", e)
                # Attempt non-strict load and print mismatches
                model_d = self.D.state_dict()
                ckpt_d = checkpoint['D_state_dict']
                matched_d = {}
                skipped_d = []
                for k, v in ckpt_d.items():
                    if k in model_d and v.shape == model_d[k].shape:
                        matched_d[k] = v
                    else:
                        skipped_d.append((k, tuple(v.shape), model_d.get(k, None).shape if k in model_d else 'not_in_model'))
                self.D.load_state_dict(matched_d, strict=False)
                print(f"[INFO] Discriminator: loaded {len(matched_d)} keys with strict=False; skipped {len(skipped_d)}")
                if skipped_d:
                    print("Discriminator skipped (key, ckpt_shape, model_shape):")
                    for s in skipped_d[:20]:
                        print(" ", s)
        else:
            print("[WARN] No 'D_state_dict' found in checkpoint.")

        self.G.eval()
        if self.D is not None:
            self.D.eval()
        self.G_weight = deepcopy(self.G.state_dict())

        # prepare latent variable and optimizer
        self.G_scheduler = LRScheduler(self.G.optim, self.args.warm_up)
        self.z_scheduler = LRScheduler(self.z_optim, self.args.warm_up)

        # loss functions
        self.ftr_net = self.D
        self.criterion = DiscriminatorLoss()

        if self.args.directed_hausdorff:
            self.directed_hausdorff = DirectedHausdorff()

        # for visualization
        self.checkpoint_pcd = []  # to save the staged checkpoints
        self.checkpoint_flags = []  # plot subtitle

        if len(args.w_D_loss) == 1:
            self.w_D_loss = args.w_D_loss * len(args.G_lrs)
        else:
            self.w_D_loss = args.w_D_loss

    def reset_G(self, pcd_id=None):
        """
        to call in every new fine_tuning
        before the 1st one also okay
        """
        self.G.load_state_dict(self.G_weight, strict=False)
        if self.args.random_G:
            self.G.train()
        else:
            self.G.eval()
        self.checkpoint_pcd = []  # to save the staged checkpoints
        self.checkpoint_flags = []
        self.pcd_id = pcd_id  # for
        if self.mask_type == 'voxel_mask':
            self.to_reset_mask = True  # reset hole center for each shape

    def set_target(self, gt=None, partial=None):
        '''
        set target
        '''
        if gt is not None:
            self.gt = gt.unsqueeze(0)
            # for visualization
            self.checkpoint_flags.append('GT')
            self.checkpoint_pcd.append(self.gt)
        else:
            self.gt = None

        if partial is not None:
            if self.args.target_downsample_method.lower() == 'fps':
                target_size = self.args.target_downsample_size
                self.target = self.downsample(partial.unsqueeze(0), target_size)
            else:
                self.target = partial.unsqueeze(0)
        else:
            self.target = self.pre_process(self.gt, stage=-1)
        # for visualization
        self.checkpoint_flags.append('target')
        self.checkpoint_pcd.append(self.target)

    def run(self, ith=-1):
        loss_dict = {}
        curr_step = 0
        count = 0
        for stage, iteration in enumerate(self.iterations):

            for i in range(iteration):
                curr_step += 1
                # setup learning rate
                self.G_scheduler.update(curr_step, self.args.G_lrs[stage])
                self.z_scheduler.update(curr_step, self.args.z_lrs[stage])

                # forward
                self.z_optim.zero_grad()

                if self.update_G_stages[stage]:
                    self.G.optim.zero_grad()

                tree = [self.z]
                x = self.G(tree)

                # masking
                x_map = self.pre_process(x, stage=stage)

                ### compute losses
                ftr_loss = self.criterion(self.ftr_net, x_map, self.target)

                dist1, dist2, _, _ = distChamfer(x_map, self.target)
                cd_loss = dist1.mean() + dist2.mean()
                # optional early stopping
                if self.args.early_stopping:
                    if cd_loss.item() < self.args.stop_cd:
                        break

                # nll corresponds to a negative log-likelihood loss
                nll = self.z**2 / 2
                nll = nll.mean()

                ### loss
                loss = ftr_loss * self.w_D_loss[stage] + nll * self.args.w_nll \
                        + cd_loss * 1

                # optional to use directed_hausdorff
                if self.args.directed_hausdorff:
                    directed_hausdorff_loss = self.directed_hausdorff(self.target, x)
                    loss += directed_hausdorff_loss * self.args.w_directed_hausdorff_loss

                # backward
                loss.backward()
                self.z_optim.step()
                if self.update_G_stages[stage]:
                    self.G.optim.step()

            # save checkpoint for each stage
            self.checkpoint_flags.append('s_' + str(stage) + ' x')
            self.checkpoint_pcd.append(x)
            self.checkpoint_flags.append('s_' + str(stage) + ' x_map')
            self.checkpoint_pcd.append(x_map)

            # test only for each stage
            if self.gt is not None:
                dist1, dist2, _, _ = distChamfer(x, self.gt)
                test_cd = dist1.mean() + dist2.mean()
                with open(self.args.log_pathname, "a") as file_object:
                    msg = str(self.pcd_id) + ',' + 'stage' + str(stage) + ',' + 'cd' + ',' + '{:6.5f}'.format(test_cd.item())
                    file_object.write(msg + '\n')

        if self.gt is not None:
            # Use .item() to get python scalar
            loss_dict = {
                'ftr_loss': float(ftr_loss.detach().cpu().item()),
                'nll': float(nll.detach().cpu().item()),
                'cd': float(test_cd.detach().cpu().item()),
            }
            self.loss_log.append(loss_dict)

        ### save point clouds
        self.x = x
        if not osp.isdir(self.args.save_inversion_path):
            os.mkdir(self.args.save_inversion_path)
        x_np = x[0].detach().cpu().numpy()
        x_map_np = x_map[0].detach().cpu().numpy()
        target_np = self.target[0].detach().cpu().numpy()
        if ith == -1:
            basename = str(self.pcd_id)
        else:
            basename = str(self.pcd_id) + '_' + str(ith)
        if self.gt is not None:
            gt_np = self.gt[0].detach().cpu().numpy()
            np.savetxt(osp.join(self.args.save_inversion_path, basename + '_gt.txt'), gt_np, fmt="%f;%f;%f")
        np.savetxt(osp.join(self.args.save_inversion_path, basename + '_x.txt'), x_np, fmt="%f;%f;%f")
        np.savetxt(osp.join(self.args.save_inversion_path, basename + '_xmap.txt'), x_map_np, fmt="%f;%f;%f")
        np.savetxt(osp.join(self.args.save_inversion_path, basename + '_target.txt'), target_np, fmt="%f;%f;%f")

        # jittering mode
        if self.args.inversion_mode == 'jittering':
            self.jitter(self.target)


    def diversity_search(self, select_y=False):
        """
        produce batch by batch
        search by 2pf and partial
        but constrainted to z dimension are large
        """
        batch_size = 50

        num_batch = int(self.select_num / batch_size)
        x_ls = []
        z_ls = []
        cd_ls = []
        tic = time.time()
        with torch.no_grad():
            for i in range(num_batch):
                z = torch.randn(batch_size, 1, 96).cuda()
                tree = [z]
                x = self.G(tree)
                dist1, dist2, _, _ = distChamfer(self.target.repeat(batch_size, 1, 1), x)
                cd = dist1.mean(1)  # single directional CD

                x_ls.append(x)
                z_ls.append(z)
                cd_ls.append(cd)

        x_full = torch.cat(x_ls)
        cd_full = torch.cat(cd_ls)
        z_full = torch.cat(z_ls)

        toc = time.time()

        cd_candidates, idx = torch.topk(cd_full, self.args.n_z_candidates, largest=False)
        z_t = z_full[idx].transpose(0, 1)
        seeds = farthest_point_sample(z_t, self.args.n_outputs).squeeze(0)
        z_ten = z_full[idx][seeds]

        self.zs = [itm.unsqueeze(0) for itm in z_ten]
        self.xs = []

    def select_z(self, select_y=False):
        tic = time.time()
        with torch.no_grad():
            if self.select_num == 0:
                self.z.zero_()
                return
            elif self.select_num == 1:
                self.z.normal_()
                return
            z_all, y_all, loss_all = [], [], []
            for i in range(self.select_num):
                z = torch.randn(1, 1, 96).to(device)
                tree = [z]
                with torch.no_grad():
                    x = self.G(tree)
                ftr_loss = self.criterion(self.ftr_net, x, self.target)
                z_all.append(z)
                # store scalar loss, not array
                loss_all.append(float(ftr_loss.detach().cpu().item()))

            toc = time.time()
            loss_all = np.array(loss_all)
            idx = np.argmin(loss_all)

            self.z.copy_(z_all[idx])
            if select_y:
                self.y.copy_(y_all[idx])

            x = self.G([self.z])

            # visualization
            if self.gt is not None:
                x_map = self.pre_process(x, stage=-1)
                dist1, dist2, _, _ = distChamfer(x, self.gt)
                cd_loss = dist1.mean() + dist2.mean()

                with open(self.args.log_pathname, "a") as file_object:
                    msg = str(self.pcd_id) + ',' + 'init' + ',' + 'cd' + ',' + '{:6.5f}'.format(cd_loss.item())
                    # print(msg)
                    file_object.write(msg + '\n')
                self.checkpoint_flags.append('init x')
                self.checkpoint_pcd.append(x)
                self.checkpoint_flags.append('init x_map')
                self.checkpoint_pcd.append(x_map)
            return z_all[idx]


    def pre_process(self, pcd, stage=-1):
        """
        transfer a pcd in the observation space:
        with the following mask_type:
            none: for ['reconstruction', 'jittering', 'morphing']
            ball_hole, knn_hole: randomly create the holes from complete pcd, similar to PF-Net
            voxel_mask: baseline in ShapeInversion
            tau_mask: baseline in ShapeInversion
            k_mask: proposed component by ShapeInversion
        """

        if self.mask_type == 'none':
            return pcd
        elif self.mask_type in ['ball_hole', 'knn_hole']:
            ### set static mask for each new partial pcd
            if self.to_reset_mask:
                # either ball hole or knn_hole, hence there might be unused configs
                self.hole_k = self.args.hole_k
                self.hole_radius = self.args.hole_radius
                self.hole_n = self.args.hole_n
                seeds = farthest_point_sample(pcd, self.hole_n)  # shape (B,hole_n)
                self.hole_centers = torch.stack([img[seed] for img, seed in zip(pcd, seeds)])  # (B, hole_n, 3)
                # turn off mask after set mask, until next partial pcd
                self.to_reset_mask = False

            ### preprocess
            flag_map = torch.ones(1, 2048, 1).cuda()
            pcd_new = pcd.unsqueeze(2).repeat(1, 1, self.hole_n, 1)
            seeds_new = self.hole_centers.unsqueeze(1).repeat(1, 2048, 1, 1)
            delta = pcd_new.add(-seeds_new)  # (B, 2048, hole_n, 3)
            dist_mat = torch.norm(delta, dim=3)
            dist_new = dist_mat.transpose(1, 2)  # (B, hole_n, 2048)

            if self.mask_type == 'knn_hole':
                # idx (B, hole_n, hole_k), dist (B, hole_n, hole_k)
                dist, idx = torch.topk(dist_new, self.hole_k, largest=False)

            for i in range(self.hole_n):
                dist_per_hole = dist_new[:, i, :].unsqueeze(2)
                if self.mask_type == 'knn_hole':
                    threshold_dist = dist[:, i, -1]
                if self.mask_type == 'ball_hole':
                    threshold_dist = self.hole_radius
                flag_map[dist_per_hole <= threshold_dist] = 0

            target = torch.mul(pcd, flag_map)
            return target
        elif self.mask_type == 'voxel_mask':
            """
            voxels in the partial and optionally surroundings are 1, the rest are 0.
            """
            ### set static mask for each new partial pcd
            if self.to_reset_mask:
                mask_partial = self.voxelize(self.target, n_bins=self.args.voxel_bins, pcd_limit=0.5, threshold=0)
                # optional to add surrounding to the mask partial
                surrounding = self.args.surrounding
                self.mask_dict = {}
                for key_gt in mask_partial:
                    x, y, z = key_gt
                    surrounding_ls = []
                    surrounding_ls.append((x, y, z))
                    for x_s in range(x - surrounding + 1, x + surrounding):
                        for y_s in range(y - surrounding + 1, y + surrounding):
                            for z_s in range(z - surrounding + 1, z + surrounding):
                                surrounding_ls.append((x_s, y_s, z_s))
                    for xyz in surrounding_ls:
                        self.mask_dict[xyz] = 1
                # turn off mask after set mask, until next partial pcd
                self.to_reset_mask = False

            ### preprocess
            n_bins = self.args.voxel_bins
            mask_tensor = torch.zeros(2048, 1)
            pcd_new = pcd * n_bins + n_bins * 0.5
            pcd_new = pcd_new.type(torch.int64)
            ls_voxels = pcd_new.squeeze(0).tolist()  # 2028 of sublists
            tuple_voxels = [tuple(itm) for itm in ls_voxels]
            for i in range(2048):
                tuple_voxel = tuple_voxels[i]
                if tuple_voxel in self.mask_dict:
                    mask_tensor[i] = 1

            mask_tensor = mask_tensor.unsqueeze(0).cuda()
            pcd_map = torch.mul(pcd, mask_tensor)
            return pcd_map
        elif self.mask_type == 'k_mask':
            pcd_map = self.k_mask(self.target, pcd, stage)
            return pcd_map
        elif self.mask_type == 'tau_mask':
            pcd_map = self.tau_mask(self.target, pcd, stage)
            return pcd_map
        else:
            raise NotImplementedError

    def voxelize(self, pcd, n_bins=32, pcd_limit=0.5, threshold=0):
        """
        given a partial/GT pcd
        return {0,1} masks with resolution n_bins^3
        voxel_limit in case the pcd is very small, but still assume it is symmetric
        threshold is needed, in case we would need to handle noise
        the form of output is a dict, key (x,y,z) , value: count
        """
        pcd_new = pcd * n_bins + n_bins * 0.5
        pcd_new = pcd_new.type(torch.int64)
        ls_voxels = pcd_new.squeeze(0).tolist()  # 2028 of sublists
        tuple_voxels = [tuple(itm) for itm in ls_voxels]
        mask_dict = {}
        for tuple_voxel in tuple_voxels:
            if tuple_voxel not in mask_dict:
                mask_dict[tuple_voxel] = 1
            else:
                mask_dict[tuple_voxel] += 1
        for voxel, cnt in list(mask_dict.items()):
            if cnt <= threshold:
                del mask_dict[voxel]
        return mask_dict

    def tau_mask(self, target, x, stage=-1):
        """
        tau mask
        """
        # dist_mat shape (B, N_target, N_output), where B = 1
        stage = max(0, stage)
        dist_tau = self.args.tau_mask_dist[stage]
        dist_mat = distChamfer_raw(target, x)
        idx0, idx1, idx2 = torch.where(dist_mat < dist_tau)
        idx = torch.unique(idx2).type(torch.long)
        x_map = x[:, idx]
        return x_map

    def k_mask(self, target, x, stage=-1):
        """
        masking based on CD.
        target: (1, N, 3), partial, can be < 2048, 2048, > 2048
        x: (1, 2048, 3)
        x_map: (1, N', 3), N' < 2048
        x_map: v1: 2048, 0 masked points
        """
        stage = max(0, stage)
        knn = self.args.k_mask_k[stage]
        if knn == 1:
            cd1, cd2, argmin1, argmin2 = distChamfer(target, x)
            idx = torch.unique(argmin1).type(torch.long)
        elif knn > 1:
            # dist_mat shape (B, 2048, 2048), where B = 1
            dist_mat = distChamfer_raw(target, x)
            # indices (B, 2048, k)
            val, indices = torch.topk(dist_mat, k=knn, dim=2, largest=False)
            # union of all the indices
            idx = torch.unique(indices).type(torch.long)

        if self.args.masking_option == 'element_product':
            mask_tensor = torch.zeros(2048, 1)
            mask_tensor[idx] = 1
            mask_tensor = mask_tensor.to(device).unsqueeze(0)
            x_map = torch.mul(x, mask_tensor)
        elif self.args.masking_option == 'indexing':
            x_map = x[:, idx]

        return x_map

    def jitter(self, x):
        z_rand = self.z.clone()

        stds = [0.3, 0.5, 0.7]
        n_jitters = 12

        flag_list = ['gt', 'recon']
        pcd_list = [self.gt, self.x]
        with torch.no_grad():
            for std in stds:

                for i in range(n_jitters):
                    z_rand.normal_()
                    z = self.z + std * z_rand
                    x_jitter = self.G([z])
                    x_np = x_jitter.squeeze(0).detach().cpu().numpy()
                    basename = '{}_std{:3.2f}_{}.txt'.format(self.pcd_id, std, i)
                    pathname = osp.join(self.args.save_inversion_path, basename)
                    np.savetxt(pathname, x_np, fmt="%f;%f;%f")
                    flag_list.append(basename)
                    pcd_list.append(x_jitter)
        self.checkpoint_pcd = pcd_list
        self.checkpoint_flags = flag_list

    def downsample(self, dense_pcd, n=2048):
        """
        input pcd cpu tensor
        return downsampled cpu tensor
        """
        idx = farthest_point_sample(dense_pcd, n)
        sparse_pcd = dense_pcd[0, idx]
        return sparse_pcd
