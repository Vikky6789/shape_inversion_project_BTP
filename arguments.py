import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from utils.common_utils import *

def str2bool(v):
    # Helper function to parse boolean CLI args, since it's used multiple times
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Arguments:
    def __init__(self, stage='pretrain'):
        self._parser = argparse.ArgumentParser(description='Arguments for pretrain|inversion|eval_treegan|eval_completion.')

        if stage == 'eval_completion':
            self.add_eval_completion_args()
        else:
            self.add_common_args()
            if stage == 'pretrain':
                self.add_pretrain_args()
            elif stage == 'inversion':
                self.add_inversion_args()
            elif stage == 'eval_treegan':
                self.add_eval_treegan_args()

    def add_common_args(self):
        ### data related
        self._parser.add_argument('--class_choice', type=str, default='chair', help='plane|cabinet|car|chair|lamp|couch|table|watercraft')
        self._parser.add_argument('--dataset', type=str, default='CRN', help='CRN|MatterPort|ScanNet|KITTI|PartNet|PFNet')
        self._parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path is required')
        self._parser.add_argument('--split', type=str, default='test', help='NOTE: train if pretrain and generate_fpd_stats; test otherwise')

        # Architecture matching checkpoint
        self._parser.add_argument('--DEGREE', type=int, default=[1,  2,   2,   2,   2,   2,   64], nargs='+', help='Upsample degrees for generator.')
        self._parser.add_argument('--G_FEAT', type=int, default=[96, 256, 256, 256, 128, 128, 128, 3], nargs='+', help='Features for generator.')
        self._parser.add_argument('--D_FEAT', type=int, default=[3, 64,  128, 256, 256, 512], nargs='+', help='Features for discriminator.')
        
        # Support for TreeGCN loop term (must be 10 to match checkpoint)
        self._parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')

        self._parser.add_argument('--loop_non_linear', default=False, type=lambda x: (str(x).lower() == 'true'))

        ### others
        self._parser.add_argument('--FPD_path', type=str, default='./evaluation/pre_statistics_chair.npz', help='Statistics file path to evaluate FPD metric. (default:all_class)')
        self._parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
        self._parser.add_argument('--ckpt_load', type=str, default='pretrained_models/chair.pt', help='Checkpoint name to load. (default:None)')

    def add_pretrain_args(self):
        ### general training related
        self._parser.add_argument('--batch_size', type=int, default=128, help='128 for cabinet, lamp, sofa, and boat; up to 512 for plane, car, chair, and table')
        self._parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs.')
        self._parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
        self._parser.add_argument('--lambdaGP', type=int, default=10, help='Lambda for GP term.')
        self._parser.add_argument('--D_iter', type=int, default=5, help='Number of iterations for discriminator.')
        self._parser.add_argument('--w_train_ls', type=float, default=[1], nargs='+', help='Train loss weightage')

        ### uniform losses related
        self._parser.add_argument('--knn_loss', default=False, type=lambda x: (str(x).lower() == 'true'))
        self._parser.add_argument('--knn_k', type=int, default=30)
        self._parser.add_argument('--knn_n_seeds', type=int, default=100)
        self._parser.add_argument('--knn_scalar', type=float, default=0.2)
        self._parser.add_argument('--krepul_loss', default=False, type=lambda x: (str(x).lower() == 'true'))
        self._parser.add_argument('--krepul_k', type=int, default=10)
        self._parser.add_argument('--krepul_n_seeds', type=int, default=20)
        self._parser.add_argument('--krepul_scalar', type=float, default=1)
        self._parser.add_argument('--krepul_h', type=float, default=0.01)
        self._parser.add_argument('--expansion_penality', default=False, type=lambda x: (str(x).lower() == 'true'))
        self._parser.add_argument('--expan_primitive_size', type=int, default=64)
        self._parser.add_argument('--expan_alpha', type=float, default=1.5)
        self._parser.add_argument('--expan_scalar', type=float, default=0.1)

        ### others
        self._parser.add_argument('--ckpt_path', type=str, default='./pretrain_checkpoints/chair', help='Checkpoint path.')
        self._parser.add_argument('--ckpt_save', type=str, default='tree_ckpt_', help='Checkpoint name to save.')
        self._parser.add_argument('--eval_every_n_epoch', type=int, default=10, help='0 means never eval')
        self._parser.add_argument('--save_every_n_epoch', type=int, default=10, help='Save models every n epochs')

    def add_inversion_args(self):
        ### loss related
        self._parser.add_argument('--w_nll', type=float, default=0.001, help='Weight for negative log-likelihood loss')
        self._parser.add_argument('--p2f_chamfer', action='store_true', default=False, help='Partial to full chamfer distance')
        self._parser.add_argument('--p2f_feature', action='store_true', default=False, help='Partial to full feature distance')
        self._parser.add_argument('--w_D_loss', type=float, default=[0.1], nargs='+', help='Discriminator feature loss weight')
        self._parser.add_argument('--directed_hausdorff', action='store_true', default=False, help='Directed hausdorff loss during inversion')
        self._parser.add_argument('--w_directed_hausdorff_loss', type=float, default=1)

        ### mask related
        self._parser.add_argument('--mask_type', type=str, default='none',
            help='none|knn_hole|ball_hole|voxel_mask|tau_mask|k_mask; use none for reconstruction, jittering, morphing; k_mask proposed for completion')
        self._parser.add_argument('--k_mask_k', type=int, default=[5, 5, 5, 5], nargs='+', help='k for k_mask (top k to keep)')
        self._parser.add_argument('--voxel_bins', type=int, default=32, help='Number of bins for voxel mask')
        self._parser.add_argument('--surrounding', type=int, default=0, help='Number of surroundings for mask v2')
        self._parser.add_argument('--tau_mask_dist', type=float, default=[0.01, 0.01, 0.01, 0.01], nargs='+', help='tau for tau_mask')
        self._parser.add_argument('--hole_radius', type=float, default=0.35, help='Radius of single hole for ball hole')
        self._parser.add_argument('--hole_k', type=int, default=500, help='k for knn ball hole')
        self._parser.add_argument('--hole_n', type=int, default=1, help='Number of holes for knn or ball hole')
        self._parser.add_argument('--masking_option', type=str, default='element_product', help='Keep zeros with element_product or remove zero with indexing')

        ### inversion mode related
        self._parser.add_argument('--inversion_mode', type=str, default='completion',
                                  help='reconstruction|completion|jittering|morphing|diversity|ball_hole_diversity|simulate_pfnet')
        self._parser.add_argument('--loss_func', type=str, default='cd', help='loss function to use: cd, emd, or dcd')
        self._parser.add_argument('--mapping_metric', type=str, default='cd', help='metric for initial shape mapping: cd, emd, or dcd')

        ### diversity
        self._parser.add_argument('--n_z_candidates', type=int, default=50, help='Number of z candidates prior to FPS')
        self._parser.add_argument('--n_outputs', type=int, default=10, help='Number of outputs for a given partial shape')

        ### other GAN inversion related
        self._parser.add_argument('--random_G', action='store_true', default=False, help='Use randomly initialized generator')
        self._parser.add_argument('--select_num', type=int, default=500, help='Number of point clouds pool to select from')
        self._parser.add_argument('--sample_std', type=float, default=1.0, help='Std dev for gaussian sampling')

        self._parser.add_argument('--iterations', type=int, default=[200, 200, 200, 200], nargs='+',
                                  help='Each sub-stage consists of 30 iterations for bulk structures, 200 for thin structures')

        self._parser.add_argument('--G_lrs', type=float, default=[2e-7, 1e-6, 1e-6, 2e-7], nargs='+', help='Learning rate steps of Generator')
        self._parser.add_argument('--z_lrs', type=float, default=[1e-2, 1e-4, 1e-4, 1e-6], nargs='+', help='Learning rate steps of latent code z')
        self._parser.add_argument('--warm_up', type=int, default=0, help='Warmup iterations')
        self._parser.add_argument('--update_G_stages', type=str2bool, default=[True, True, True, True], nargs='+', help='Update G control at each stage')
        self._parser.add_argument('--progressive_finetune', action='store_true', default=False, help='Progressive finetune at each stage')
        self._parser.add_argument('--init_by_p2f_chamfer', action='store_true', default=False, help='Init by partial-to-full chamfer')
        self._parser.add_argument('--early_stopping', action='store_true', default=False, help='Early stopping')
        self._parser.add_argument('--stop_cd', type=float, default=0.0005, help='CD threshold for stopping')
        self._parser.add_argument('--target_downsample_method', default="", type=str, help='Optional downsampling method via FPS')
        self._parser.add_argument('--target_downsample_size', default=1024, type=int, help='Target downsample size by FPS')

        ### others
        self._parser.add_argument('--save_inversion_path', default='', help='Directory to save generated point clouds')
        self._parser.add_argument('--dist', action='store_true', default=False, help='Train with distributed implementation')
        self._parser.add_argument('--port', type=str, default='12345', help='Port id for distributed training')
        self._parser.add_argument('--visualize', action='store_true', default=False, help='Enable visualization')
        
        #saving intermediate steps
        self._parser.add_argument('--save_interval', type=int, default=10,help='Save intermediate reconstructions every N iterations')

    def add_eval_completion_args(self):
        self._parser.add_argument('--eval_with_GT', type=str2bool, default=False, help='Eval with ground truth on real scans')
        self._parser.add_argument('--saved_results_path', type=str, required=True, help='Path of saved results for evaluation')

    def add_eval_treegan_args(self):
        self._parser.add_argument('--eval_treegan_mode', type=str, default='FPD', help='MMD|FPD|save|generate_fpd_stats')
        self._parser.add_argument('--save_sample_path', type=str, required=True, help='Dir to save generated point clouds')
        self._parser.add_argument('--model_pathname', type=str, required=True, help='Model pathname to evaluate')
        self._parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
        self._parser.add_argument('--n_samples', type=int, default=5000, help='Number of points generated by G')

    def parser(self):
        return self._parser
