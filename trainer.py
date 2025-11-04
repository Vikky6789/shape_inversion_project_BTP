# ==========================================================
# trainer.py (Kaggle-Compatible Version)
# ==========================================================

import sys
import os
import time
from collections import OrderedDict
import random

# -------------------------
# Fix import paths for Kaggle
# -------------------------
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, "data"))
sys.path.append(os.path.join(repo_root, "utils"))
sys.path.append(os.path.join(repo_root, "model"))
sys.path.append(os.path.join(repo_root, "external"))

import torch
import torch.distributed as dist
import torch.optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader, DistributedSampler

# -------------------------
# Project Imports
# -------------------------
from data.CRN_dataset import CRNShapeNet
from data.ply_dataset import PlyDataset
from arguments import Arguments
from utils.pc_transform import voxelize
from utils.plot import draw_any_set
from utils.common_utils import *
from utils.inversion_dist import *
from loss import *

from shape_inversion import ShapeInversion
from model.treegan_network import Generator, Discriminator
from external.ChamferDistancePytorch.chamfer_python import distChamfer, distChamfer_raw


# ==========================================================
# Trainer Class
# ==========================================================
class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Setup distributed params
        if self.args.dist:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank, self.world_size = 0, 1

        self.inversion_mode = args.inversion_mode

        save_inversion_dirname = args.save_inversion_path.split('/')
        log_pathname = './logs/' + save_inversion_dirname[-1] + '.txt'
        args.log_pathname = log_pathname

        # Ensure checkpoint attribute exists
        if not hasattr(self.args, 'checkpoint'):
            self.args.checkpoint = getattr(self.args, 'ckpt_load', None)

        # Ensure args.device exists
        if not hasattr(self.args, 'device'):
            self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -------------------------
        # Model creation
        # -------------------------
        self.model = ShapeInversion(self.args)
        if self.inversion_mode == 'morphing':
            self.model2 = ShapeInversion(self.args)
            self.model_interp = ShapeInversion(self.args)

        # -------------------------
        # Dataset loader
        # -------------------------
        if self.args.dataset == 'ply':
            dataset = PlyDataset(self.args)
        else:
            dataset = CRNShapeNet(self.args)

        sampler = DistributedSampler(dataset) if self.args.dist else None

        if self.inversion_mode == 'morphing':
            self.dataloader = DataLoader(
                dataset, batch_size=2, shuffle=False, sampler=sampler,
                num_workers=1, pin_memory=False)
        else:
            self.dataloader = DataLoader(
                dataset, batch_size=1, shuffle=False, sampler=sampler,
                num_workers=1, pin_memory=False)

    # ==========================================================
    # Main run entry
    # ==========================================================
    def run(self):
        if self.inversion_mode in ['reconstruction', 'completion', 'jittering', 'simulate_pfnet']:
            self.train()
        elif self.inversion_mode in ['diversity', 'ball_hole_diversity']:
            self.train_diversity()
        elif self.inversion_mode == 'morphing':
            self.train_morphing()
        else:
            raise NotImplementedError

    # ==========================================================
    # Training Modes
    # ==========================================================
    def train(self):
        for i, data in enumerate(self.dataloader):
            tic = time.time()

            # Dataset check
            if self.args.dataset == 'ply':
                partial, index = data
                gt = None
            else:
                gt, partial, index = data
                gt = gt.squeeze(0).to(self.args.device)

            if self.args.inversion_mode == 'simulate_pfnet':
                n_removal = 512
                choice = [torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]
                chosen = random.sample(choice, 1)[0]
                dist_val = torch.norm(gt.add(-chosen.to(self.args.device)), dim=1)
                top_dist, idx = torch.topk(dist_val, k=2048 - n_removal)
                partial = gt[idx]

            partial = partial.squeeze(0).to(self.args.device)
            self.model.reset_G(pcd_id=index.item())

            if partial is None or self.args.inversion_mode in [
                'reconstruction', 'jittering', 'morphing', 'ball_hole', 'knn_hole'
            ]:
                self.model.set_target(gt=gt)
            else:
                self.model.set_target(gt=gt, partial=partial)

            self.model.select_z(select_y=False)
            self.model.run()

            toc = time.time()
            if self.rank == 0:
                print(i, 'out of', len(self.dataloader), 'done in', int(toc - tic), 's')

            if self.args.visualize:
                pcd_list = self.model.checkpoint_pcd
                flag_list = self.model.checkpoint_flags
                output_dir = self.args.save_inversion_path + '_visual'
                output_stem = str(index.item())
                if self.args.inversion_mode == 'jittering':
                    draw_any_set(flag_list, pcd_list, output_dir, output_stem, layout=(4, 10))
                else:
                    draw_any_set(flag_list, pcd_list, output_dir, output_stem, layout=(3, 4))

        print('<<<<<<<<<<<< rank', self.rank, 'completed >>>>>>>>>>>>>>')


    def train_diversity(self):
        for i, data in enumerate(self.dataloader):
            tic = time.time()
            if self.args.dataset == 'ply':
                partial, index = data
                gt = None
            else:
                gt, partial, index = data
                gt = gt.squeeze(0).to(self.args.device)

            if self.args.inversion_mode == 'ball_hole_diversity':
                pcd = gt.unsqueeze(0).clone()
                self.hole_radius = self.args.hole_radius
                self.hole_n = self.args.hole_n
                seeds = farthest_point_sample(pcd, self.hole_n)
                self.hole_centers = torch.stack([itm[seed] for itm, seed in zip(pcd, seeds)])
                flag_map = torch.ones(1, 2048, 1).to(self.args.device)
                pcd_new = pcd.unsqueeze(2).repeat(1, 1, self.hole_n, 1)
                seeds_new = self.hole_centers.unsqueeze(1).repeat(1, 2048, 1, 1)
                delta = pcd_new.add(-seeds_new)
                dist_mat = torch.norm(delta, dim=3)
                dist_new = dist_mat.transpose(1, 2)
                for ii in range(self.hole_n):
                    dist_per_hole = dist_new[:, ii, :].unsqueeze(2)
                    threshold_dist = self.hole_radius
                    flag_map[dist_per_hole <= threshold_dist] = 0
                partial = torch.mul(pcd, flag_map).squeeze(0)
                norm = torch.norm(partial, dim=1)
                idx = torch.where(norm > 0)
                partial = partial[idx].to(self.args.device)
                print(index.item(), 'partial shape', partial.shape)
            else:
                partial = partial.squeeze(0).to(self.args.device)

            self.model.reset_G(pcd_id=index.item())
            self.model.set_target(gt=gt, partial=partial)
            self.model.diversity_search()

            pcd_ls = [gt.unsqueeze(0), partial.unsqueeze(0)]
            flag_ls = ['gt', 'input']
            for ith, z in enumerate(self.model.zs):
                self.model.reset_G(pcd_id=index.item())
                self.model.set_target(gt=gt, partial=partial)
                self.model.z.data = z.data
                self.model.run(ith=ith)
                self.model.xs.append(self.model.x)
                flag_ls.append(str(ith))
                pcd_ls.append(self.model.x)

            if self.args.visualize:
                output_stem = str(index.item())
                output_dir = self.args.save_inversion_path + '_visual'
                if self.args.n_outputs <= 10:
                    layout = (3, 4)
                elif self.args.n_outputs <= 20:
                    layout = (4, 6)
                else:
                    layout = (6, 9)
                draw_any_set(flag_ls, pcd_ls, output_dir, output_stem, layout=layout)
            if self.rank == 0:
                toc = time.time()
                print(i, 'out of', len(self.dataloader), 'done in', int(toc - tic), 's')
            print(f"{i} / {len(self.dataloader)} completed")


    def train_morphing(self):
        for i, data in enumerate(self.dataloader):
            tic = time.time()
            gt, partial, index = data
            gt = gt.to(self.args.device)

            # Reconstruction on both pcs
            self.model.reset_G(pcd_id=index[0].item())
            self.model.set_target(gt=gt[0])
            self.model.select_z(select_y=False)
            self.model.run()

            self.model2.reset_G(pcd_id=index[1].item())
            self.model2.set_target(gt=gt[1])
            self.model2.select_z(select_y=False)
            self.model2.run()

            # Interpolation
            interpolated_pcd = []
            interpolated_flag = []
            weight1 = self.model.G.state_dict()
            weight2 = self.model2.G.state_dict()
            weight_interp = OrderedDict()
            with torch.no_grad():
                for jj in range(11):
                    alpha = jj / 10
                    z_interp = alpha * self.model.z + (1 - alpha) * self.model2.z
                    for k, w1 in weight1.items():
                        w2 = weight2[k]
                        weight_interp[k] = alpha * w1 + (1 - alpha) * w2
                    self.model_interp.G.load_state_dict(weight_interp)
                    x_interp = self.model_interp.G([z_interp])
                    interpolated_pcd.append(x_interp)
                    interpolated_flag.append(str(alpha))

            if self.args.visualize:
                pcd_ls = [gt[0].unsqueeze(0)] + interpolated_pcd + [gt[1].unsqueeze(0)]
                flag_ls = ['gt_1'] + interpolated_flag + ['gt_2']
                output_dir = self.args.save_inversion_path + '_visual'
                output_stem = str(index[0].item()) + '_' + str(index[1].item())
                draw_any_set(flag_ls, pcd_ls, output_dir, output_stem, layout=(3, 6))

                output_dir2 = self.args.save_inversion_path + '_interpolates'
                if not os.path.isdir(output_dir2):
                    os.mkdir(output_dir2)
                for flag, pcd in zip(flag_ls, pcd_ls):
                    pcd = pcd.squeeze(0).detach().cpu().numpy()
                    np.savetxt(os.path.join(output_dir2, output_stem + '_' + flag + '.txt'), pcd, fmt="%f;%f;%f")

            if self.rank == 0:
                toc = time.time()
                print(i, 'out of', len(self.dataloader), 'done in', int(toc - tic), 's')


# ==========================================================
# Main Entry Point
# ==========================================================
if __name__ == "__main__":
    args = Arguments(stage='inversion').parser().parse_args()
    args.device = torch.device('cuda:' + str(args.gpu) if args.gpu != -1 and torch.cuda.is_available() else 'cpu')

    if not os.path.isdir('./logs/'):
        os.mkdir('./logs/')
    if not os.path.isdir('./saved_results'):
        os.mkdir('./saved_results')

    if args.dist:
        dist_init(args.port)

    trainer = Trainer(args)
    trainer.run()
