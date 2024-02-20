print("strawberry")

# loss function and metrics
from trustworthai.utils.losses_and_metrics.dice_loss import DiceLossWithWeightedEmptySlices
from trustworthai.utils.losses_and_metrics.dice_loss_metric import DiceLossMetric, SsnDiceMeanMetricWrapper

# predefined training dataset
from trustworthai.utils.data_preprep.dataset_pipelines import load_data
from torch.utils.data import ConcatDataset

# fitter
from trustworthai.utils.fitting_and_inference.fitters.basic_lightning_fitter import StandardLitModelWrapper
from trustworthai.utils.fitting_and_inference.fitters.p_unet_fitter import PUNetLitModelWrapper
from trustworthai.utils.fitting_and_inference.get_trainer import get_trainer

# model
from trustworthai.journal_run.model_load.load_ssn import load_ssn
from trustworthai.journal_run.model_load.load_punet import load_p_unet
from trustworthai.journal_run.model_load.load_deterministic import load_deterministic
from trustworthai.journal_run.model_load.load_evidential import load_evidential
from trustworthai.models.stochastic_wrappers.ssn.LowRankMVCustom import LowRankMultivariateNormalCustom
from trustworthai.models.stochastic_wrappers.ssn.ReshapedDistribution import ReshapedDistribution

# optimizer and lr scheduler
import torch


import numpy as np
from tqdm import tqdm
import scipy.stats
from trustworthai.utils.plotting.saving_plots import save, imsave
from trustworthai.utils.print_and_write_func import print_and_write

# misc
import argparse
import os
import shutil
import shlex
from collections import defaultdict
from tqdm import tqdm
import sys
from natsort import natsorted

import pandas as pd
from trustworthai.analysis.connected_components.connected_comps_2d import conn_comp_2d_analysis
from trustworthai.analysis.evaluation_metrics.challenge_metrics import getAVD, getDSC, getHausdorff, getLesionDetection, do_challenge_metrics
from sklearn import metrics
import math

import torch
import matplotlib.pyplot as plt
from trustworthai.utils.plotting.saving_plots import save
from trustworthai.utils.print_and_write_func import print_and_write
from trustworthai.analysis.calibration.helper_funcs import *
from tqdm import tqdm
from trustworthai.utils.logits_to_preds import normalize_samples

# data
from trustworthai.utils.data_preprep.dataset_pipelines import load_clinscores_data, load_data, ClinScoreDataRetriever
from trustworthai.utils.uncertainty_maps.entropy_map import entropy_map_from_samples


# evaluation code
from trustworthai.journal_run.evaluation.new_scripts.eval_helper_functions import *
from trustworthai.journal_run.evaluation.new_scripts.model_predictions import *
from trustworthai.analysis.connected_components.connected_comps_2d import *


from trustworthai.utils.data_preprep.dataset_pipelines import load_clinscores_data, load_data, ClinScoreDataRetriever

from torch.utils.data import ConcatDataset
from twaidata.torchdatasets.DirectoryParser3DMRIDataset import *

print("banana")

MODEL_LOADERS = {
    "deterministic":load_deterministic,
    "mc_drop":load_deterministic,
    "evidential":load_evidential,
    "ssn":load_ssn,
    "punet":load_p_unet,
}

MODEL_OUTPUT_GENERATORS = {
    "deterministic":deterministic_mean,
    "mc_drop":mc_drop_mean_and_samples,
    "evidential":evid_mean,
    "ssn":ssn_mean_and_samples,
    "punet":punet_mean_and_samples,
    "ind":ssn_mean_and_samples,
    "ens":ensemble_mean_and_samples,
    "ssn_ens":ssn_ensemble_mean_and_samples,
}

VOXELS_TO_WMH_RATIO = 382
VOXELS_TO_WMH_RATIO_EXCLUDING_EMPTY_SLICES = 140

uncertainty_thresholds = torch.arange(0, 0.7, 0.01)

def UIRO(pred, thresholded_umap, seg1, seg2):
    IR = (seg1 != seg2)
    error = (seg1 == seg2) * (seg1 != pred)
    thresholded_umap[error] = 0
    return fast_dice(thresholded_umap, IR)


def JUEO(pred, thresholded_umap, seg1, seg2):
    IR = (seg1 != seg2)
    error = (seg1 == seg2) * (seg1 != pred)
    thresholded_umap[IR] = 0
    return fast_dice(thresholded_umap, error)

def per_rater_UEO(pred, thresholded_umap, seg1, seg2):
    error1 = (seg1 != pred)
    error2 = (seg2 != pred)
    
    return fast_dice(thresholded_umap, error1), fast_dice(thresholded_umap, error2)

def per_threshold_ueos(means, ent_maps, rater0, rater1, xs3d_test):
    uiro_curves = []
    jueo_curves = []
    for i in tqdm(range(len(xs3d_test))):
        uiro = []
        jueo = []
        m = means[i].argmax(dim=1).cuda()
        for t in uncertainty_thresholds:
            e = ent_maps[i] > t
            e = e.cuda()
            uiro.append(UIRO(m, e.clone(), rater0[i].cuda(), rater1[i].cuda()))
            jueo.append(JUEO(m, e.clone(), rater0[i].cuda(), rater1[i].cuda()))
        uiro_curves.append(uiro)
        jueo_curves.append(jueo)
        
    return uiro_curves, jueo_curves

def get_2rater_rmse(pred, y0, y1, p=0.1):
    label = torch.zeros(pred.shape, device='cuda')
    label[:,0] = (y0 == 0) & (y1 == 0)
    label[:,1] = (y0 == 1) & (y1 == 1)
    diff = (y0 != y1)
    label[:,0][diff] = 0.5
    label[:, 1][diff] = 0.5
    
    locs = pred[:,1] > p
    # print(pred.shape)
    
    pred = pred.moveaxis(1, -1)[locs]
    label = label.moveaxis(1, -1)[locs]
    
    rmse = ((pred - label).square().sum(dim=1) / pred.shape[1]).mean().sqrt()

    return rmse.item()

def get_IR_rmse(pred, y0, y1, p=0.1):
    label = torch.zeros(pred.shape, device='cuda')
    label[:,0] = (y0 == 0) & (y1 == 0)
    label[:,1] = (y0 == 1) & (y1 == 1)
    diff = (y0 != y1)
    label[:,0][diff] = 0.5
    label[:, 1][diff] = 0.5
    
    locs = diff
    # print(pred.shape)
    
    pred = pred.moveaxis(1, -1)[locs]
    label = label.moveaxis(1, -1)[locs]
    
    rmse = ((pred - label).square().sum(dim=1) / pred.shape[1]).mean().sqrt()

    return rmse.item()

import torch
import torch.nn.functional as F

def dilate(tensor, kernel_size=3, iterations=1):
    """
    Dilate a 3D binary tensor using a cubic kernel.
    
    Parameters:
    - tensor: A 3D binary tensor of shape (C, H, W, D) where C is the channel (1 for binary images).
    - kernel_size: Size of the cubic kernel for dilation.
    - iterations: Number of times dilation is applied.
    
    Returns:
    - Dilated tensor.
    """
    padding = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=tensor.device)
    for _ in range(iterations):
        tensor = F.conv3d(tensor, kernel, padding=padding, groups=1)
    return torch.clamp(tensor, 0, 1)

def erode(tensor, kernel_size=3, iterations=1):
    """
    Erode a 3D binary tensor using a cubic kernel.
    
    Parameters:
    - tensor: A 3D binary tensor of shape (C, H, W, D).
    - kernel_size: Size of the cubic kernel for erosion.
    - iterations: Number of times erosion is applied.
    
    Returns:
    - Eroded tensor.
    """
    padding = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=tensor.device)
    for _ in range(iterations):
        tensor = F.conv3d(1 - tensor, kernel, padding=padding, groups=1)
    return 1 - torch.clamp(tensor, 0, 1)

def find_edges(tensor, kernel_size=3):
    """
    Find the inner and outer edges of a segmentation in a 3D binary tensor.
    
    Parameters:
    - tensor: A 3D binary tensor of shape (H, W, D).
    - kernel_size: Size of the cubic kernel for dilation and erosion.
    
    Returns:
    - inner_edges: The inner edges of the segmentation.
    - outer_edges: The outer edges of the segmentation.
    """
    tensor = tensor.unsqueeze(0)
    dilated = dilate(tensor, kernel_size)
    eroded = erode(tensor, kernel_size)
    outer_edges = dilated - tensor
    inner_edges = tensor - eroded
    
    outer_edges = outer_edges.squeeze()
    inner_edges = inner_edges.squeeze()
    
    return inner_edges, outer_edges


def get_rmse_stats(means, rater0, rater1):
    rmses = []
    IR_rmses = []
    p = 0.1
    for i in tqdm(range(len(means))):
        y0 = rater0[i]
        y1 = rater1[i]
        m = means[i].cuda()
        m = m.softmax(dim=1)
        rmse = get_2rater_rmse(m, y0, y1, p)
        ir_rmse = get_IR_rmse(m, y0, y1, p)

        rmses.append(rmse)
        IR_rmses.append(ir_rmse)
        
    return rmses, IR_rmses
    
def edge_deducted_UIRO(pred, thresholded_umap, seg1, seg2):
    pred = pred.type(torch.float32).cuda()
    inner_edge, outer_edge = find_edges(pred)#.unsqueeze(0))
    seg1 = seg1.cuda()
    seg2 = seg2.cuda()
    
    IR = (seg1 != seg2)
    error = (seg1 == seg2) * (seg1 != pred)
    thresholded_umap[error] = 0
    
    non_edge_locs = ((1-inner_edge) * (1 - outer_edge)) == 1
    thresholded_umap *= non_edge_locs
    IR *= non_edge_locs
    
    return fast_dice(thresholded_umap, IR)


def edge_deducted_JUEO(pred, thresholded_umap, seg1, seg2):
    pred = pred.type(torch.float32).cuda()
    inner_edge, outer_edge = find_edges(pred)#.unsqueeze(0))
    seg1 = seg1.cuda()
    seg2 = seg2.cuda()
    
    IR = (seg1 != seg2)
    error = (seg1 == seg2) * (seg1 != pred)
    thresholded_umap[IR] = 0
    
    non_edge_locs = ((1-inner_edge) * (1 - outer_edge)) == 1
    thresholded_umap *= non_edge_locs
    error *= non_edge_locs
    
    return fast_dice(thresholded_umap, error)

def per_threshold_edge_deducted_ueos(means, ent_maps, rater0, rater1, xs3d_test):
    uiro_curves = []
    jueo_curves = []
    for i in tqdm(range(len(xs3d_test))):
        uiro = []
        jueo = []
        m = means[i].argmax(dim=1).cuda()
        for t in uncertainty_thresholds:
            e = ent_maps[i] > t
            e = e.cuda()
            uiro.append(edge_deducted_UIRO(m, e.clone(), rater0[i].cuda(), rater1[i].cuda()))
            jueo.append(edge_deducted_JUEO(m, e.clone(), rater0[i].cuda(), rater1[i].cuda()))
        uiro_curves.append(uiro)
        jueo_curves.append(jueo)
        
    return uiro_curves, jueo_curves

def soft_dice(pred, target):
    
    numerator = 2 * (pred * target).sum()
    denominator = (target**2).sum() + (pred**2).sum()
    
    return (numerator / denominator).item()


def soft_edge_deducted_UIRO(pred, umap, seg1, seg2):
    pred = pred.type(torch.float32).cuda()
    inner_edge, outer_edge = find_edges(pred)#.unsqueeze(0))
    seg1 = seg1.cuda()
    seg2 = seg2.cuda()
    
    IR = (seg1 != seg2)
    error = (seg1 == seg2) * (seg1 != pred)
    umap[error] = 0
    
    non_edge_locs = ((1-inner_edge) * (1 - outer_edge)) == 1
    umap *= non_edge_locs
    IR *= non_edge_locs
    
    return soft_dice(umap, IR)


def soft_edge_deducted_JUEO(pred, umap, seg1, seg2):
    pred = pred.type(torch.float32).cuda()
    inner_edge, outer_edge = find_edges(pred)#.unsqueeze(0))
    seg1 = seg1.cuda()
    seg2 = seg2.cuda()
    
    IR = (seg1 != seg2)
    error = (seg1 == seg2) * (seg1 != pred)
    umap[IR] = 0
    
    non_edge_locs = ((1-inner_edge) * (1 - outer_edge)) == 1
    umap *= non_edge_locs
    error *= non_edge_locs
    
    return soft_dice(umap, error)

def soft_UIRO(pred, umap, seg1, seg2):
    IR = (seg1 != seg2)
    error = (seg1 == seg2) * (seg1 != pred)
    umap[error] = 0
    return soft_dice(umap, IR)


def soft_JUEO(pred, umap, seg1, seg2):
    IR = (seg1 != seg2)
    error = (seg1 == seg2) * (seg1 != pred)
    umap[IR] = 0
    return soft_dice(umap, error)

def soft_per_rater_UEO(pred, umap, seg1, seg2):
    error1 = (seg1 != pred)
    error2 = (seg2 != pred)
    
    return soft_dice(umap, error1), soft_dice(umap, error2)

def soft_ueo_metrics(means, ent_maps, rater0, rater1, xs3d_test):
    sUIRO = []
    sJUEO = []
    sUEO_r1 = []
    sUEO_r2 = []
    s_ed_UIRO = []
    s_ed_JUEO = []
    for i in tqdm(range(len(xs3d_test))):
        m = means[i].argmax(dim=1).cuda()
        e = ent_maps[i].cuda()
        y0, y1 = rater0[i].cuda(), rater1[i].cuda()
        
        sUIRO.append(soft_UIRO(m, e.clone(), y0, y1))
        sJUEO.append(soft_JUEO(m, e.clone(), y0, y1))
        s_ed_UIRO.append(soft_edge_deducted_UIRO(m, e.clone(), y0, y1))
        s_ed_JUEO.append(soft_edge_deducted_JUEO(m, e.clone(), y0, y1))
        
        sueo1, sueo2 = soft_per_rater_UEO(m, e, y0, y1)
        sUEO_r1.append(sueo1)
        sUEO_r2.append(sueo2)
        
    return sUIRO, sJUEO, sUEO_r1, sUEO_r2, s_ed_UIRO, s_ed_JUEO

def conn_comp_analysis(means, ent_maps, rater0, rater1):
    
    ind_entirely_uncert = []
    ind_proportion_uncertain = []
    ind_mean_uncert = []
    ind_sizes = []

    for i in tqdm(range(len(means))):
        pred = means[i].cuda().argmax(dim=1)
        e = ent_maps[i].cuda()
        y0 = rater0[i]
        y1 = rater1[i]

        disagreement = (y0 != y1)

        ccs = cc3d.connected_components(disagreement.type(torch.int32).numpy(), connectivity=26) # 26-connected
        ccs = torch.from_numpy(ccs.astype(np.float32)).cuda()

        entirely_uncertain = [[] for _ in range(len(uncertainty_thresholds))]
        proportion_uncertain = [[] for _ in range(len(uncertainty_thresholds))]
        mean_uncert = []
        sizes = []
        
        if len(ccs.unique()) != 1:
            for cc_id in ccs.unique():
                if cc_id == 0:
                    continue
                cc = ccs == cc_id
                size = cc.sum().item()
                sizes.append(size)
                mean_uncert.append(e[cc].mean().item())

                for j, t in enumerate(uncertainty_thresholds):
                    et = e > t
                    uncert_cc_sum = (cc * et).sum().item()
                    proportion_uncertain[j].append(uncert_cc_sum / size)
                    entirely_uncertain[j].append(uncert_cc_sum == size)

        ind_entirely_uncert.append(entirely_uncertain)
        ind_proportion_uncertain.append(proportion_uncertain)
        ind_mean_uncert.append(mean_uncert)
        ind_sizes.append(sizes)
        
    return ind_entirely_uncert, ind_proportion_uncertain, ind_mean_uncert, ind_sizes

def pixelwise_metrics(means, ent_maps, rater0, rater1, xs3d_test):
    JTP = []
    JFP = []
    JFN = []
    IR = []
    for i in tqdm(range(len(xs3d_test))):
        m = means[i].argmax(dim=1).cuda().type(torch.long)
        e = ent_maps[i].cuda()
        y0, y1 = rater0[i].cuda().type(torch.long), rater1[i].cuda().type(torch.long)
        
        # flatten for indexing
        e = e.view(-1)
        m = m.view(-1)
        y0 = y0.view(-1)
        y1 = y1.view(-1)
        
        joint = (y0 == y1)
        ir = (y0 != y1)
        
        JTP.append(e[(joint * y0 * m)==1].cpu())
        JFP.append(e[(joint * (1 - y0) * m)==1].cpu())
        JFN.append(e[(joint * y0 * (1 - m))==1].cpu())
        IR.append(e[ir].cpu())
        
    return JTP, JFP, JFN, IR

def edge_deducted_pixelwise_metrics(means, ent_maps, rater0, rater1, xs3d_test):
    JTP = []
    JFP = []
    JFN = []
    IR = []
    for i in tqdm(range(len(xs3d_test))):
        m = means[i].argmax(dim=1).cuda().type(torch.long)
        inner_edge, outer_edge = find_edges(m.type(torch.float32))
        e = ent_maps[i].cuda()
        y0, y1 = rater0[i].cuda().type(torch.long), rater1[i].cuda().type(torch.long)
        
        # flatten for indexing
        e = e.view(-1)
        m = m.view(-1)
        y0 = y0.view(-1)
        y1 = y1.view(-1)
        
        # non edge area
        non_edge = (1-inner_edge) * (1-outer_edge)
        non_edge = non_edge.view(-1)
        
        joint = (y0 == y1) * non_edge
        ir = (y0 != y1) * (non_edge == 1)
        
        JTP.append(e[(joint * y0 * m)==1].cpu())
        JFP.append(e[(joint * (1 - y0) * m)==1].cpu())
        JFN.append(e[(joint * y0 * (1 - m))==1].cpu())
        IR.append(e[ir].cpu())
        
    return JTP, JFP, JFN, IR

# make sure that we have the volume difference per individual
# make sure that this collection is put by the samples that are collected by volume!!!!!
def vd_dist_and_skew(samples, rater0, rater1):
    vds_rater0 = []
    vds_rater1 = []
    vds_rater_mean = []
    sample_vol_skew = []
    for i, s in tqdm(enumerate(samples), total=len(samples)):
        y0 = rater0[i].cuda().sum().item()
        y1 = rater1[i].cuda().sum().item()
        y_mean = ( y0 + y1 ) / 2
        s = s.cuda().argmax(dim=2)

        vds_rater0.append([(((sj.sum() - y0) / y0) * 100).item() for sj in s])
        vds_rater1.append([(((sj.sum() - y1) / y1) * 100).item() for sj in s])
        vds_mean = [(((sj.sum() - y_mean) / y_mean) * 100).item() for sj in s]
        vds_rater_mean.append(vds_mean)
        sample_vol_skew.append(scipy.stats.skew(np.array(vds_mean), bias=True))
    
    return vds_rater0, vds_rater1, vds_rater_mean, sample_vol_skew

def fast_iou(pred, target):
    p1 = (pred == 1)
    t1 = (target == 1)
    intersection = (pred == 1) & (target == 1)
    numerator = intersection.sum()
    denominator = p1.sum() + t1.sum() - numerator
    return (numerator/(denominator + 1e-30)).item()

def individual_multirater_iou_GED(mean, rater_ys, sample):
    ged = 0
    ys = [r for r in rater_ys]
    ss = sample.cuda().argmax(dim=2)
    num_samples = ss.shape[0]

    dists_ab = 0
    count_ab = 0
    for s in ss:
        for y in ys:
            pred = s#.argmax(dim=1)
            dists_ab += (1 - fast_iou(pred, y.cuda()))
            # print(dists_ab)
            # print(s.shape)
            count_ab += 1

    dists_ab /= count_ab # num_samples # count should be num_samples * num_raters for consistent number of raters but Ive just done this count for now.
    dists_ab *= 2

    dists_aa = 0
    count_aa = 0
    for j, y1 in enumerate(ys):
        for k, y2 in enumerate(ys):
            if j == k:
                continue
            dists_aa += (1 - fast_iou(y1.cuda(), y2.cuda()))
            count_aa += 1

    dists_aa /= count_aa

    dists_bb = 0
    for j, s1 in enumerate(ss):
        for k, s2 in enumerate(ss):
            if j == k:
                continue
            dists_bb += (1 - fast_iou(s1, s2))

    dists_bb /= (num_samples * (num_samples - 1))

    ged = dists_ab - dists_aa - dists_bb
        
    return ged

def multirater_iou_GED(means, rater_ys, samples):
    geds = []
    
    for i in tqdm(range(len(means)), position=0, leave=True):
        ys = [r[i] for r in rater_ys]
        ss = samples[i].cuda().argmax(dim=2)
        num_samples = ss.shape[0]
        
        dists_ab = 0
        count_ab = 0
        for s in ss:
            for y in ys:
                pred = s#.argmax(dim=1)
                dists_ab += (1 - fast_iou(pred, y.cuda()))
                # print(dists_ab)
                # print(s.shape)
                count_ab += 1
        
        dists_ab /= count_ab # num_samples # count should be num_samples * num_raters for consistent number of raters but Ive just done this count for now.
        dists_ab *= 2
        
        dists_aa = 0
        count_aa = 0
        for j, y1 in enumerate(ys):
            for k, y2 in enumerate(ys):
                if j == k:
                    continue
                dists_aa += (1 - fast_iou(y1.cuda(), y2.cuda()))
                count_aa += 1
        
        dists_aa /= count_aa
        
        dists_bb = 0
        for j, s1 in enumerate(ss):
            for k, s2 in enumerate(ss):
                if j == k:
                    continue
                dists_bb += (1 - fast_iou(s1, s2))
        
        dists_bb /= (num_samples * (num_samples - 1))
        
        ged = dists_ab - dists_aa - dists_bb
        if not np.isnan(ged):
            geds.append(ged)
        #break
        
    return torch.Tensor(geds)

def construct_parser():
    parser = argparse.ArgumentParser(description = "train models")
    
    # folder arguments
    parser.add_argument('--ckpt_dir', default='s2208943/results/revamped_models/', type=str)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--repo_dir', default=None, type=str)
    parser.add_argument('--result_dir', default=None, type=str)
    parser.add_argument('--eval_split', default='val', type=str)
    
    # data generation arguments
    parser.add_argument('--dataset', default='MSS3', type=str)
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--test_split', default=0.15, type=float)
    parser.add_argument('--val_split', default=0.15, type=float)
    parser.add_argument('--empty_slice_retention', default=0.1, type=float)
    
    # select the model type to evaluate
    parser.add_argument('--model_type', default="deterministic", type=str)
    parser.add_argument('--uncertainty_type', default="deterministic", type=str)
    parser.add_argument('--eval_sample_num', default=10, type=int)
    
    # general arguments for the loss function
    parser.add_argument('--loss_name', default='dice+xent', type=str)
    parser.add_argument('--dice_factor', default=1, type=float) # 5
    parser.add_argument('--xent_factor', default=1, type=float) # 0.01
    parser.add_argument('--xent_reweighting', default=None, type=float)
    parser.add_argument('--xent_weight', default="none", type=str)
    parser.add_argument('--dice_empty_slice_weight', default=0.5, type=float)
    parser.add_argument('--tversky_beta', default=0.7, type=float)
    parser.add_argument('--reduction', default='mean_sum', type=str)
    
    # evidential arguments
    parser.add_argument('--kl_factor', default=0.1, type=float)
    parser.add_argument('--kl_anneal_count', default=452*4, type=int)
    parser.add_argument('--use_mle', default=0, type=int)
    parser.add_argument('--analytic_kl', default=0, type=int)
    
    # p-unet arguments
    parser.add_argument('--kl_beta', default=10.0, type=float)
    parser.add_argument('--use_prior_for_dice', default="false", type=str)
    parser.add_argument('--punet_sample_dice_coeff', default=0.05, type=float)
    parser.add_argument('--latent_dim', default=12, type=int)
    
    # ssn arguments
    parser.add_argument('--ssn_rank', default=15, type=int)
    parser.add_argument('--ssn_epsilon', default=1e-5, type=float)
    parser.add_argument('--ssn_mc_samples', default=10, type=int)
    parser.add_argument('--ssn_sample_dice_coeff', default=0.05, type=float)
    parser.add_argument('--ssn_pre_head_layers', default=16, type=int)
    
    # training paradigm arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--dropout_p', default=0.0, type=float)
    parser.add_argument('--encoder_dropout1', default=0, type=int)
    parser.add_argument('--encoder_dropout2', default=0, type=int)
    parser.add_argument('--decoder_dropout1', default=0, type=int)
    parser.add_argument('--decoder_dropout2', default=0, type=int)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--early_stop_patience', default=15, type=int)
    parser.add_argument('--scheduler_type', default='step', type=str)
    parser.add_argument('--scheduler_step_size', default=1000, type=int)
    parser.add_argument('--scheduler_gamma', default=0.5, type=float)
    parser.add_argument('--scheduler_power', default=0.9, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--cross_validate', default="true", type=str)
    parser.add_argument('--cv_split', default=0, type=int)
    parser.add_argument('--cv_test_fold_smooth', default=1, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--overwrite', default="false", type=str)
    parser.add_argument('--no_test_fold', default='false', type=str)
    
    return parser

def main(args):
    # sanitise arguments
    args.overwrite = True if args.overwrite.lower() == "true" else False
    args.cross_validate = True if args.cross_validate.lower() == "true" else False
    args.use_prior_for_dice = True if args.use_prior_for_dice.lower() == "true" else False
    print(f"CHECKPOINT DIR: {args.ckpt_dir}")
    
    if args.dataset == "chal" and args.eval_split == "all":
        args.cv_split = 0 #  we are evaluating on the whole dataset anyway and the dataset doesn't divide into more than 5 folds with the parameters used on the ed dataset.
    #print(args)
    
    uncertainty_thresholds = torch.arange(0, 0.7, 0.01)
    
    # check if folder exists
    # model_result_folder = os.path.join(args.repo_dir, args.result_dir)
    # if not args.overwrite:
    #     existing_files = os.listdir(model_result_folder)
    #     for f in existing_files:
    #         if args.model_name + "_" in f:
    #             raise ValueError(f"ovewrite = false and model results exist! folder={model_result_folder}, model_name={args.model_name}")
    # with open(os.path.join(model_result_folder, f"{args.model_name}_init.txt"), "w") as f:
    #                       f.write("generating results\n")
        
    # setup xent reweighting factor
    XENT_VOXEL_RESCALE = VOXELS_TO_WMH_RATIO - (1-args.empty_slice_retention) * (VOXELS_TO_WMH_RATIO - VOXELS_TO_WMH_RATIO_EXCLUDING_EMPTY_SLICES)

    XENT_WEIGHTING = XENT_VOXEL_RESCALE/2
    args.xent_reweighting = XENT_WEIGHTING
    
    # load the model
    print("LOADING INITIAL MODEL")
    model_dir = os.path.join(args.ckpt_dir, args.model_name)  
    print("model dir: ", model_dir)
    model_raw, loss, val_loss = MODEL_LOADERS[args.model_type](args)
    model = load_best_checkpoint(model_raw, loss, model_dir, punet=args.model_type == "punet")
        
    if args.dataset.lower() == "mss3":
        ds = MSS3InterRaterDataset()
        xs3d_test = []
        ys3d_test = []

        for (xs, ys, ind) in tqdm(ds):
            if "wmhes" in ys.keys() and "wmhmvh" in ys.keys():
                xs3d_test.append(torch.stack([xs['FLAIR'], xs['mask'], xs['T1']], dim=0))
                ys3d_test.append(torch.stack([ys['wmhes'], ys['wmhmvh']], dim=0))
    elif args.dataset.lower() == "lbc":
        ds = LBCInterRaterDataset()
        xs3d_test = []
        ys3d_test = []

        for (xs, ys, ind) in tqdm(ds):
            if "wmh_flthresh" in ys.keys() and "wmh" in ys.keys():
                xs3d_test.append(torch.stack([xs['FLAIR'], xs['mask'], xs['T1']], dim=0))
                ys3d_test.append(torch.stack([ys['wmh_flthresh'], ys['wmh']], dim=0))
    elif args.dataset.lower() == "challenge":
        ds = WMHChallengeInterRaterDataset()
        xs3d_test = []
        ys3d_test = []

        for (xs, ys, ind) in tqdm(ds):
            if "wmho3" in ys.keys() and "wmho4" in ys.keys():
                xs3d_test.append(torch.stack([xs['FLAIR'], xs['mask'], xs['T1']], dim=0))
                ys3d_test.append(torch.stack([ys['wmho3'], ys['wmho4']], dim=0))
    
    # configuring raters
    rater0 = [y[0] for y in ys3d_test]
    rater1 = [y[1] for y in ys3d_test]
    gt_vols = [(torch.sum(y[0]).item(), torch.sum(y[1]).item()) for y in ys3d_test]
    mean_gt_vols = [(torch.sum(y[0]).item() + torch.sum(y[1]).item())/2 for y in ys3d_test]
    
    rater0_ds = list(zip(xs3d_test, rater0))

    ### results collection loop below
    raters = [rater0, rater1]
    rater_results = [defaultdict(lambda : {}) for _ in range(len(raters))]
    overall_results = defaultdict(lambda: {})
    pixelwise_results = defaultdict(lambda: {})

    print("loading model predictions")
    ns_init = 10 if args.uncertainty_type == "ens" else 30
    means, samples_all, misc = get_means_and_samples(model_raw, rater0_ds, num_samples=ns_init, model_func=MODEL_OUTPUT_GENERATORS[args.uncertainty_type], args=args)

    rmses, IR_rmses = get_rmse_stats(means, rater0, rater1)

    all_result_ns = [2, 10, 30]
    
    for num_samples in [2, 3, 5, 7, 10, 15, 20, 25, 30]:
        overall_results[num_samples][f'rmses'] = rmses
        overall_results[num_samples][f'IR_rmses'] = IR_rmses
        print("NUM SAMPLES: ", num_samples)
        args.eval_sample_num = num_samples
        try:
            # load the predictions
            print("extracting sample subset")
            print(args.uncertainty_type)
            samples = [s[:num_samples] for s in tqdm(samples_all)]
            print("computing uncertainty maps")
            ent_maps = get_uncertainty_maps(means, samples, misc, args)

            # run the evaluation on the samplesUIRO_curves
            print("GETTING PER SAMPLE RESULTS")
            if samples[0] is not None:
                samples = [reorder_samples(s) for s in samples]
                for r, rater in enumerate(raters):
                    sample_top_dices, sample_dices = per_sample_metric(samples, rater, f=fast_dice, do_argmax=True, do_softmax=False, minimise=False)
                    sample_best_avds, sample_avds = per_sample_metric(samples, rater, f=fast_avd, do_argmax=True, do_softmax=False, minimise=True)
                    # sample_best_rmses, sample_rmses = per_sample_metric(samples, ys3d_test, f=fast_rmse, do_argmax=False, do_softmax=True, minimise=True)

                    # rater_results[r][num_samples]['sample_top_dice'] = sample_top_dices
                    # rater_results[r][num_samples]['sample_best_avd'] = sample_best_avds
                    overall_results[num_samples][f'rater{r}_sample_top_dice'] = sample_top_dices
                    overall_results[num_samples][f'rater{r}_sample_best_avd'] = sample_best_avds

                # ged by volume
                print("COMPUTING GED BY VOLUME")
                overall_results[num_samples]['GED_vol_sorted'] = multirater_iou_GED(means, raters, samples)

                # UEO metrics
                if num_samples in all_result_ns:
                    print("UEO metrics curves")
                    uiro_curves, jueo_curves = per_threshold_ueos(means, ent_maps, rater0, rater1, xs3d_test)
                    uiro_curves = torch.Tensor(uiro_curves)
                    jueo_curves = torch.Tensor(jueo_curves)
                    for ti, t in enumerate(uncertainty_thresholds):
                        overall_results[num_samples][f'UIRO_curves_t{t:.2f}'] = uiro_curves[:,ti]
                        overall_results[num_samples][f'JUEO_curves_t{t:.2f}'] = jueo_curves[:,ti]

                    no_edge_uiro_curves, no_edge_jueo_curves = per_threshold_edge_deducted_ueos(means, ent_maps, rater0, rater1, xs3d_test)
                    no_edge_uiro_curves = torch.Tensor(no_edge_uiro_curves)
                    no_edge_jueo_curves = torch.Tensor(no_edge_jueo_curves)
                    for ti, t in enumerate(uncertainty_thresholds):
                        overall_results[num_samples][f'no_edge_uiro_curves_t{t:.2f}'] = no_edge_uiro_curves[:,ti]
                        overall_results[num_samples][f'no_edge_jueo_curves_t{t:.2f}'] = no_edge_jueo_curves[:,ti]

                    print("soft UEO metrics values")
                    sUIRO, sJUEO, sUEO_r1, sUEO_r2, s_ed_UIRO, s_ed_JUEO = soft_ueo_metrics(means, ent_maps, rater0, rater1, xs3d_test)
                    overall_results[num_samples]['sUIRO'] = sUIRO
                    overall_results[num_samples]['sJUEO'] = sJUEO
                    overall_results[num_samples]['sUEO_r1'] = sUEO_r1
                    overall_results[num_samples]['sUEO_r2'] = sUEO_r2
                    overall_results[num_samples]['s_ed_UIRO'] = s_ed_UIRO
                    overall_results[num_samples]['s_ed_JUEO'] = s_ed_JUEO

                    print("connected component analysis")
                    ind_entirely_uncert, ind_proportion_uncertain, ind_mean_uncert, ind_sizes = conn_comp_analysis(means, ent_maps, rater0, rater1)
                    mean_ind_entirely_uncert = torch.stack([torch.Tensor([torch.Tensor(tr).mean() for tr in indr]) for indr in ind_entirely_uncert])
                    mean_ind_proportion_uncertain = torch.stack([torch.Tensor([torch.Tensor(tr).mean() for tr in indr]) for indr in ind_proportion_uncertain])
                    mean_ind_mean_uncert = torch.stack([torch.Tensor(indr).mean() for indr in ind_entirely_uncert])

                    for ti, t in enumerate(uncertainty_thresholds):
                        overall_results[num_samples][f'mean_ind_entirely_uncert_t{t:.2f}'] = mean_ind_entirely_uncert[:,ti]
                        overall_results[num_samples][f'mean_ind_proportion_uncertain_t{t:.2f}'] = mean_ind_proportion_uncertain[:,ti]
                    overall_results[num_samples]['mean_ind_mean_uncert'] = mean_ind_mean_uncert
                    # overall_results[num_samples]['ind_sizes'] = ind_sizes

                    print("pixelwise analysis")
                    JTP, JFP, JFN, IR = pixelwise_metrics(means, ent_maps, rater0, rater1, xs3d_test)
                    edJTP, edJFP, edJFN, edIR = edge_deducted_pixelwise_metrics(means, ent_maps, rater0, rater1, xs3d_test)
                    pixelwise_results[num_samples]['JTP'] = JTP
                    pixelwise_results[num_samples]['JFP'] = JFP
                    pixelwise_results[num_samples]['JFN'] = JFN
                    pixelwise_results[num_samples]['IR'] = IR
                    pixelwise_results[num_samples]['edJTP'] = edJTP 
                    pixelwise_results[num_samples]['edJFP'] = edJFP
                    pixelwise_results[num_samples]['edJFN'] = edJFN
                    pixelwise_results[num_samples]['edIR'] = edIR

                    print("volume difference distribution information")
                    vds_rater0, vds_rater1, vds_rater_mean, sample_vol_skew = vd_dist_and_skew(samples, rater0, rater1)
                    vds_rater0 = torch.Tensor(vds_rater0)
                    vds_rater1 = torch.Tensor(vds_rater1)
                    vds_rater_mean = torch.Tensor(vds_rater_mean)
                    for ns in range(num_samples):
                        overall_results[num_samples][f'vds_rater0_sample{ns}'] = vds_rater0[:,ns]
                        overall_results[num_samples][f'vds_rater1_sample{ns}'] = vds_rater1[:,ns]
                        overall_results[num_samples][f'vds_rater_mean_sample{ns}'] = vds_rater_mean[:,ns]
                    overall_results[num_samples]['sample_vol_skew'] = sample_vol_skew

                # best dice when sorting the sample for dice
                print("best dice and GED results sorted by dice")
                overall_results[num_samples]['GED_dice_sorted'] = []
                for r, rater in enumerate(raters):
                    best_dice = []
                    for i, s in tqdm(enumerate(samples)):                
                        y = rater[i].cuda()
                        s = reorder_samples_by_dice(s, y)
                        best_dice.append(fast_dice(s[-1].cuda().argmax(dim=1), y))
                        overall_results[num_samples][f'rater{r}_best_dice_dsorted_ss{num_samples}'] = best_dice

                        if r == 0:
                            # ged score by dice
                            overall_results[num_samples]['GED_dice_sorted'].append(individual_multirater_iou_GED(means[i], [r[i] for r in raters], s))
                    
                overall_results[ns]['mean_gt_vols'] = mean_gt_vols
                
                path = "/home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/interrater_experiments/results/"
                for key in overall_results[num_samples].keys():
                    print(key, len(overall_results[num_samples][key]))
                pd.DataFrame(overall_results[num_samples]).to_csv(path + f"inter_rater_{args.dataset}_{args.uncertainty_type}_cv{args.cv_split}_ns{num_samples}.csv")
                np.savez(path + f"voxelwise_IRstats_{args.dataset}_{args.uncertainty_type}_cv{args.cv_split}_ns{num_samples}.npz", pixelwise_results, allow_pickle=True)

                # TODO I SHOULD DO GED BASED ON DICE sorting AND VOLUME sorting
            else:
                print("samples is None, breaking now")
                break

        except Exception as e:
            print(f"failed at {num_samples}")
            print(e)
            raise e
            break

if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
