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
from trustworthai.run.model_load.load_ssn import load_ssn
from trustworthai.run.model_load.load_punet import load_p_unet
from trustworthai.run.model_load.load_deterministic import load_deterministic
from trustworthai.models.stochastic_wrappers.ssn.LowRankMVCustom import LowRankMultivariateNormalCustom
from trustworthai.models.stochastic_wrappers.ssn.ReshapedDistribution import ReshapedDistribution

# optimizer and lr scheduler
import torch


import numpy as np
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

print("banana")

from trustworthai.utils.data_preprep.dataset_pipelines import load_clinscores_data, load_data, ClinScoreDataRetriever

models_folder = "/home/s2208943/ipdis/results/cross_validated_models/"

def construct_parser():
    parser = argparse.ArgumentParser(description = "train models")
    
    # folder arguments
    parser.add_argument('--ckpt_dir', default='s2208943/results/revamped_models/', type=str)
    parser.add_argument('--model_name', default=None, type=str)
    
    # data generation arguments
    parser.add_argument('--dataset', default='ed', type=str)
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--test_split', default=0.15, type=float)
    parser.add_argument('--val_split', default=0.15, type=float)
    parser.add_argument('--empty_slice_retention', default=0.1, type=float)
    
    # model specific parameters SSN
    parser.add_argument('--ssn_rank', default=15, type=int)
    parser.add_argument('--ssn_epsilon', default=1e-5, type=float)
    parser.add_argument('--ssn_mc_samples', default=10, type=int)
    parser.add_argument('--ssn_sample_dice_coeff', default=0.05, type=float)
    parser.add_argument('--ssn_pre_head_layers', default=16, type=int)
    
    # evidential loss parameters
    parser.add_argument('--kl_factor', default=0.1, type=float)
    parser.add_argument('--kl_anneal_count', default=452*4, type=int)
    
     # model specific parameters Punet
    parser.add_argument('--kl_beta', default=10.0, type=float)
    parser.add_argument('--use_prior_for_dice', default=False, type=bool)
    parser.add_argument('--punet_sample_dice_coeff', default=0.05, type=float)
    parser.add_argument('--latent_dim', default=12, type=int)
    
    # general arguments for the loss function
    parser.add_argument('--dice_factor', default=5, type=float)
    parser.add_argument('--xent_factor', default=0.01, type=float)
    parser.add_argument('--dice_empty_slice_weight', default=0.5, type=float)
    
    # general arguments for the loss function
    parser.add_argument('--loss_name', default='dice+xent', type=str)
    # parser.add_argument('--dice_factor', default=5, type=float)
    # parser.add_argument('--xent_factor', default=0.01, type=float)
    parser.add_argument('--xent_weight', default='none', type=str)
    #parser.add_argument('--dice_empty_slice_weight', default=0.5, type=float)
    parser.add_argument('--tversky_beta', default=0.7, type=float)
    parser.add_argument('--reduction', default='mean_sum', type=str)
    
    # training paradigm arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--dropout_p', default=0.0, type=float)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--early_stop_patience', default=15, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--cross_validate', default=False, type=bool)
    parser.add_argument('--cv_split', default=0, type=int)
    parser.add_argument('--cv_test_fold_smooth', default=1, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--overwrite', default=False, type=bool)
    
    return parser

def load_best_checkpoint(model, loss, model_ckpt_folder, punet=False):
    # this is ultimately going to need to be passed a model wrapper when I implement P-Unet....
    
    # the path to the best checkpoint is stored as a single line in a txt file along with each model
    with open(os.path.join(model_ckpt_folder, "best_ckpt.txt"), "r") as f:
        ckpt_file = os.path.join(model_ckpt_folder, f.readlines()[0][:-1].split("/")[-1])
    
    if punet:
        return PUNetLitModelWrapper.load_from_checkpoint(ckpt_file, model=model, loss=loss, 
                                    logging_metric=lambda : None)
    return StandardLitModelWrapper.load_from_checkpoint(ckpt_file, model=model, loss=loss, 
                                    logging_metric=lambda : None)


def entropy_map_from_mean(mean, do_normalize=True):
    "samples is of shape samples, batch size, channels, image dims  [s, b, c *<dims>]"
    if mean.shape[1] == 1:
        raise ValueError("not implemented for implicit background class")
    else:
        assert mean.shape[1] == 2
    
    if do_normalize:
        probs = torch.nn.functional.softmax(mean, dim=1)
    else:
        probs = mean
    ent_map = torch.sum(-probs * torch.log(probs+1e-30), dim=1)

    return ent_map

def generate_means_and_samples_SSN_Ens(splits=6, dataset_stride=2, temp=1, num_samples=10):
    # load data
    clin_retriever = ClinScoreDataRetriever()
    
    # load model
    class TestArgs():
        def __init__(self, ):
            args_dict = {
                "dropout_p":0,
                "ssn_pre_head_layers":32,
                "ssn_rank":15,
                "ssn_epsilon":1e-5,
                "dice_empty_slice_weight":0.5,
                "ssn_mc_samples":10,
                "dice_factor":5,
                "xent_factor":0.01,
                "ssn_sample_dice_coeff":0.05
            }

            for key, value in args_dict.items():
                setattr(self, key, value)

    args = TestArgs()

    model_raw, loss = load_ssn(args)
    model_raw = model_raw.cuda()
    
    models_folder = "/home/s2208943/ipdis/results/cross_validated_models/"
    #model_names = os.listdir(models_folder)

    model_base_name = "ssn_cross_validate_32_"

    model_outs = defaultdict(lambda : defaultdict(lambda : {'means':[], 'samples':[]}))

    test_datasets = []

    for split in range(splits):
        # load specific data split
        train_ds_clin, val_ds_clin, test_ds_clin = clin_retriever.load_clinscores_data(
            combine_all=False,
            test_proportion=0.15, 
            validation_proportion=0.15,
            seed=3407,
            cross_validate=True,
            cv_split=split,
            cv_test_fold_smooth=1,
        )
        test_datasets.append(test_ds_clin)
        print("size: ", len(test_ds_clin))


        for ens in range(10):
            model_name = model_base_name + f"split{split}_ens{ens}/"
            model_path = models_folder + model_name

            # with open(model_path + "best_ckpt.txt") as f:
            #     lines = f.readlines()
            #     args_lines = [l[:-1].split(": ") for l in lines[1:]]
            #     args_lines = [f'--{l[0]} {l[1]}' for l in args_lines]
            #     args_line = " ".join(args_lines)
            #     parser = construct_parser()
            #     args = parser.parse_args(shlex.split(args_line))

            # load the model
            model = load_best_checkpoint(model_raw, loss, model_path)
            model.eval()


            dataskip = dataset_stride
            # means = []
            # samples = []
            for i, data in enumerate(tqdm(test_ds_clin, position=0, leave=True)):
                if i % dataskip == 0:
                    x = data[0]
                    with torch.no_grad():
                        mean, sample = model.mean_and_sample(x.swapaxes(0,1).cuda(), num_samples=num_samples // 10, temperature=temp, symmetric=False)

                        model_outs[split][i]['means'].append(mean.cpu())
                        model_outs[split][i]['samples'].append(sample.cpu())

                        # means.append(mean.cpu())
                        # samples.append(sample.cpu())

            #model_outs[split][ens] = {'means':means, 'samples':samples}

    for split in model_outs.keys():
        for idx in tqdm(model_outs[split].keys(), position=0, leave=True):
            model_outs[split][idx]['means'] = torch.stack(model_outs[split][idx]['means'], dim=0).mean(dim=0)
            model_outs[split][idx]['samples'] = torch.cat(model_outs[split][idx]['samples'], dim=0)

    means = [model_outs[split][idx]['means'] for split in model_outs.keys() for idx in model_outs[split].keys()]
    samples = [model_outs[split][idx]['samples'] for split in model_outs.keys() for idx in model_outs[split].keys()]

    return means, samples, ConcatDataset(test_datasets)


def generate_means_and_samples_SSN_Ens_Mean(splits=6, dataset_stride=2, temp=1, num_samples=10, components=10):
    # load data
    print("loading")
    clin_retriever = ClinScoreDataRetriever()
    print("loaded")
    
    test_datasets = []

    # load model
    class TestArgs():
        def __init__(self, ):
            args_dict = {
                "dropout_p":0,
                "ssn_pre_head_layers":32,
                "ssn_rank":15,
                "ssn_epsilon":1e-5,
                "dice_empty_slice_weight":0.5,
                "ssn_mc_samples":10,
                "dice_factor":5,
                "xent_factor":0.01,
                "ssn_sample_dice_coeff":0.05
            }

            for key, value in args_dict.items():
                setattr(self, key, value)

    args = TestArgs()

    model_raw, loss = load_ssn(args)
    model_raw = model_raw.cuda()
    model_raw.return_cpu_dist = True

    models_folder = "/home/s2208943/ipdis/results/cross_validated_models/"
    #model_names = os.listdir(models_folder)

    model_base_name = "ssn_cross_validate_32_"

    means = []
    samples = []

    for split in range(splits):
        # load specific data split
        train_ds_clin, val_ds_clin, test_ds_clin = clin_retriever.load_clinscores_data(
            combine_all=False,
            test_proportion=0.15, 
            validation_proportion=0.12,
            seed=3407,
            cross_validate=True,
            cv_split=split,
            cv_test_fold_smooth=1,
        )
        test_datasets.append(test_ds_clin)
        # print("size: ", len(test_ds_clin))

        dataskip = dataset_stride

        for i, data in enumerate(tqdm(test_ds_clin, position=0, leave=True)):
            if i % dataskip == 0:
                x = data[0].swapaxes(0,1).cuda()
                distribution_means = []
                distribution_cov_diags = []
                distribution_cov_factors = []
                distribution_event_shapes = []

                for ens in range(components):
                    # print(ens)
                    model_name = model_base_name + f"split{split}_ens{ens}/"
                    model_path = models_folder + model_name
                    model = load_best_checkpoint(model_raw, loss, model_path)
                    model.eval()

                    with torch.no_grad():
                        mean, cov_diag, cov_factor, event_shape = model(x)
                    distribution_means.append(mean.cpu())
                    distribution_cov_diags.append(cov_diag.cpu())
                    distribution_cov_factors.append(cov_factor.cpu())
                    distribution_event_shapes.append(event_shape)

                # print(distribution_means[0].shape)

                distribution_means = torch.stack(distribution_means, dim=0).mean(dim=0)
                distribution_cov_diags = torch.stack(distribution_cov_diags, dim=0).mean(dim=0)
                distribution_cov_factors = torch.stack(distribution_cov_factors, dim=0).mean(dim=0)

                # print(distribution_means.shape)

                dist = LowRankMultivariateNormalCustom(distribution_means, distribution_cov_factors, distribution_cov_diags)
                dist = ReshapedDistribution(dist, distribution_event_shapes[0])

                means.append((dist.mean / temp).cpu())
                samples.append((model_raw._samples_from_dist(dist, num_samples=num_samples)/temp).cpu())

    return means, samples, ConcatDataset(test_datasets)

def generate_means_and_samples_SSN(splits=6, dataset_stride=2, temp=1, num_samples=10, independent=False):
    # load data
    clin_retriever = ClinScoreDataRetriever()
    
    rank = 15
    if independent:
        rank = 1
    
    # load model
    class TestArgs():
        def __init__(self, ):
            args_dict = {
                "dropout_p":0,
                "ssn_pre_head_layers":32,
                "ssn_rank":rank,
                "ssn_epsilon":1e-5,
                "dice_empty_slice_weight":0.5,
                "ssn_mc_samples":10,
                "dice_factor":5,
                "xent_factor":0.01,
                "ssn_sample_dice_coeff":0.05
            }

            for key, value in args_dict.items():
                setattr(self, key, value)

    args = TestArgs()

    model_raw, loss = load_ssn(args)
    model_raw = model_raw.cuda()
    
    models_folder = "/home/s2208943/ipdis/results/cross_validated_models/"
    #model_names = os.listdir(models_folder)

    if independent:
        model_base_name = "ssn_ind_32_cross_validate_"
    else:
        model_base_name = "ssn_cross_validate_32_"
    ensemble_element = 1

    means = []
    samples = []

    test_datasets = []

    for split in range(splits):
        # load specific data split
        train_ds_clin, val_ds_clin, test_ds_clin = clin_retriever.load_clinscores_data(
            combine_all=False,
            test_proportion=0.15, 
            validation_proportion=0.12,
            seed=3407,
            cross_validate=True,
            cv_split=split,
            cv_test_fold_smooth=1,
        )
        test_datasets.append(test_ds_clin)
        print("size: ", len(test_ds_clin))


        if independent:
            model_name = model_base_name + f"split{split}/"
        else:
            model_name = model_base_name + f"split{split}_ens{ensemble_element}/"
        model_path = models_folder + model_name

        model = load_best_checkpoint(model_raw, loss, model_path)
        model.eval()

        dataskip = dataset_stride
        for i, data in enumerate(tqdm(test_ds_clin, position=0, leave=True)):
            if i % dataskip == 0:
                x = data[0]
                with torch.no_grad():
                    mean, sample = model.mean_and_sample(x.swapaxes(0,1).cuda(), num_samples=num_samples, temperature=temp)
                    means.append(mean.cpu())
                    samples.append(sample.cpu())

    return means, samples, ConcatDataset(test_datasets)


def generate_means_and_samples_Ensemble(splits=6, dataset_stride=2, temp=1, num_samples=10):
    print("strawberry")
    # load data
    clin_retriever = ClinScoreDataRetriever()
    
    models_folder = "/home/s2208943/ipdis/results/cross_validated_models/"
    #model_names = os.listdir(models_folder)

    model_outs = defaultdict(lambda : defaultdict(lambda : {'samples':[]}))

    test_datasets = []

    for split in range(splits):
        # load specific data split
        train_ds_clin, val_ds_clin, test_ds_clin = clin_retriever.load_clinscores_data(
            combine_all=False,
            test_proportion=0.15, 
            validation_proportion=0.12,
            seed=3407,
            cross_validate=True,
            cv_split=split,
            cv_test_fold_smooth=1,
        )
        test_datasets.append(test_ds_clin)
        print("size: ", len(test_ds_clin))


        for ens in range(10):
            model_name = f"deterministic_ens{ens}_dice+xent_split{split}/"
            model_path = models_folder + model_name

            with open(model_path + "best_ckpt.txt") as f:
                lines = f.readlines()
                args_lines = [l[:-1].split(": ") for l in lines[1:]]
                args_lines = [f'--{l[0]} {l[1]}' for l in args_lines]
                args_line = " ".join(args_lines)
                parser = construct_parser()
                args = parser.parse_args(shlex.split(args_line))
                
            model_raw, loss = load_deterministic(args)

            # load the model
            model = load_best_checkpoint(model_raw, loss, model_path)
            model.eval()



            dataskip = dataset_stride
            # means = []
            # samples = []
            for i, data in enumerate(tqdm(test_ds_clin, position=0, leave=True)):
                if i % dataskip == 0:
                    x = data[0]
                    with torch.no_grad():
                        sample = model(x.swapaxes(0,1).cuda()).cpu()
                        sample = sample / temp
                        model_outs[split][i]['samples'].append(sample.cpu())

    for split in model_outs.keys():
        for idx in tqdm(model_outs[split].keys(), position=0, leave=True):
            model_outs[split][idx]['samples'] = torch.stack(model_outs[split][idx]['samples'], dim=0)
            model_outs[split][idx]['means'] = model_outs[split][idx]['samples'].mean(dim=0)

    means = [model_outs[split][idx]['means'] for split in model_outs.keys() for idx in model_outs[split].keys()]
    samples = [model_outs[split][idx]['samples'] for split in model_outs.keys() for idx in model_outs[split].keys()]

    return means, samples, ConcatDataset(test_datasets)

def generate_means_and_samples_MC_Dropout(splits=6, dataset_stride=2, temp=1, num_samples=10):
    print("strawberry")
    # load data
    clin_retriever = ClinScoreDataRetriever()
    
    models_folder = "/home/s2208943/ipdis/results/cross_validated_models/"
    #model_names = os.listdir(models_folder)

    means = []
    samples = []

    test_datasets = []

    for split in range(splits):
        # load specific data split
        train_ds_clin, val_ds_clin, test_ds_clin = clin_retriever.load_clinscores_data(
            combine_all=False,
            test_proportion=0.15, 
            validation_proportion=0.12,
            seed=3407,
            cross_validate=True,
            cv_split=split,
            cv_test_fold_smooth=1,
        )
        test_datasets.append(test_ds_clin)
        print("size: ", len(test_ds_clin))
        
        model_name = f"mcdropout_dice+xent_split{split}/"
        model_path = models_folder + model_name
        with open(model_path + "best_ckpt.txt") as f:
            lines = f.readlines()
            args_lines = [l[:-1].split(": ") for l in lines[1:]]
            args_lines = [f'--{l[0]} {l[1]}' for l in args_lines]
            args_line = " ".join(args_lines)
            parser = construct_parser()
            args = parser.parse_args(shlex.split(args_line))

        model_raw, loss = load_deterministic(args)
        print(args.dropout_p)

        # load the model
        model = load_best_checkpoint(model_raw, loss, model_path)

        with open(model_path + "best_ckpt.txt") as f:
            lines = f.readlines()
            args_lines = [l[:-1].split(": ") for l in lines[1:]]
            args_lines = [f'--{l[0]} {l[1]}' for l in args_lines]
            args_line = " ".join(args_lines)
            parser = construct_parser()
            args = parser.parse_args(shlex.split(args_line))

        model_raw, loss = load_deterministic(args)

        # load the model
        model = load_best_checkpoint(model_raw, loss, model_path)

        dataskip = dataset_stride
        for i, data in enumerate(tqdm(test_ds_clin, position=0, leave=True)):
            if i % dataskip == 0:
                x = data[0]
                with torch.no_grad():
                    model.eval()
                    mean = model(x.swapaxes(0,1).cuda()).cpu()
                    means.append(mean / temp)
                    ind_samples = []
                    model.train()
                    for j in range(num_samples):
                        sample = model(x.swapaxes(0,1).cuda()).cpu()
                        sample = sample / temp
                        ind_samples.append(sample)
                    samples.append(torch.stack(ind_samples, dim=0))

    return means, samples, ConcatDataset(test_datasets)

def generate_means_and_samples_PUNet(splits=6, dataset_stride=2, temp=1, num_samples=10, prior=True, on_samples=False):
    # load data
    clin_retriever = ClinScoreDataRetriever()
    
    models_folder = "/home/s2208943/ipdis/results/cross_validated_models/"
    #model_names = os.listdir(models_folder)

    model_base_name = "punet_lt_"
    if prior:
        typename = "prior"
    else:
        typename = "posterior"
        
    if on_samples:
        samplesname = "on_samples_"
    else:
        samplesname = ""
        
    ensemble_element = 0

    means = []
    samples = []

    test_datasets = []

    for split in range(splits):
        # load specific data split
        train_ds_clin, val_ds_clin, test_ds_clin = clin_retriever.load_clinscores_data(
            combine_all=False,
            test_proportion=0.15, 
            validation_proportion=0.12,
            seed=3407,
            cross_validate=True,
            cv_split=split,
            cv_test_fold_smooth=1,
        )
        test_datasets.append(test_ds_clin)
        print("size: ", len(test_ds_clin))

        model_name = f"{model_base_name}{typename}_dice_{samplesname}split{split}/"
        model_path = models_folder + model_name
        
        with open(model_path + "best_ckpt.txt") as f:
            lines = f.readlines()
            args_lines = [l[:-1].split(": ") for l in lines[1:]]
            args_lines = [f'--{l[0]} {l[1]}' for l in args_lines]
            args_line = " ".join(args_lines)
            parser = construct_parser()
            args = parser.parse_args(shlex.split(args_line))
        
        model_raw, loss = load_p_unet(args)
        model_raw = model_raw.cuda()

        model = load_best_checkpoint(model_raw, loss, model_path, punet=True)
        model.eval()

        dataskip = dataset_stride
        for i, data in enumerate(tqdm(test_ds_clin, position=0, leave=True)):
            if i % dataskip == 0:
                x = data[0]
                y = data[1]
                with torch.no_grad():
                    model_raw(x.swapaxes(0,1).cuda(), y.cuda(), training=False)
                    mean = model_raw.sample(use_prior_mean=True).cpu() / temp
                    means.append(mean)
                    
                    ind_samples = []
                    for j in range(num_samples):
                        ind_samples.append(model_raw.sample(testing=False).cpu() / temp)
                    
                    samples.append(torch.stack(ind_samples, dim=0))

    return means, samples, ConcatDataset(test_datasets)

import torch.distributions as td

def generate_means_and_samples_CategoricalSoftmax(splits=6, dataset_stride=2, temp=1, num_samples=10, loss_name='dice+xent'):
    print("strawberry")
    # load data
    clin_retriever = ClinScoreDataRetriever()
    
    models_folder = "/home/s2208943/ipdis/results/cross_validated_models/"
    #model_names = os.listdir(models_folder)

    means = []
    samples = []

    test_datasets = []

    for split in range(splits):
        # load specific data split
        train_ds_clin, val_ds_clin, test_ds_clin = clin_retriever.load_clinscores_data(
            combine_all=False,
            test_proportion=0.15, 
            validation_proportion=0.12,
            seed=3407,
            cross_validate=True,
            cv_split=split,
            cv_test_fold_smooth=1,
        )
        test_datasets.append(test_ds_clin)
        print("size: ", len(test_ds_clin))
        
        model_name = f"deterministic_ens0_{loss_name}_split{split}/"
        model_path = models_folder + model_name
        with open(model_path + "best_ckpt.txt") as f:
            lines = f.readlines()
            args_lines = [l[:-1].split(": ") for l in lines[1:]]
            args_lines = [f'--{l[0]} {l[1]}' for l in args_lines]
            args_line = " ".join(args_lines)
            parser = construct_parser()
            args = parser.parse_args(shlex.split(args_line))

        model_raw, loss = load_deterministic(args)
        print(args.dropout_p)

        # load the model
        model = load_best_checkpoint(model_raw, loss, model_path)

        with open(model_path + "best_ckpt.txt") as f:
            lines = f.readlines()
            args_lines = [l[:-1].split(": ") for l in lines[1:]]
            args_lines = [f'--{l[0]} {l[1]}' for l in args_lines]
            args_line = " ".join(args_lines)
            parser = construct_parser()
            args = parser.parse_args(shlex.split(args_line))

        model_raw, loss = load_deterministic(args)

        # load the model
        model = load_best_checkpoint(model_raw, loss, model_path)

        dataskip = dataset_stride
        for i, data in enumerate(tqdm(test_ds_clin, position=0, leave=True)):
            if i % dataskip == 0:
                x = data[0]
                with torch.no_grad():
                    model.eval()
                    mean = model(x.swapaxes(0,1).cuda()).cpu() / temp
                    means.append(mean)
                    dist = td.Categorical(probs = torch.softmax(mean, dim=1).moveaxis(1,-1))
                    ind_samples = []
                    samples.append(dist.sample((num_samples,)))
                    
    samples2 = []
    for s in tqdm(samples, position=0, leave=True):
        new_shape = list(s.unsqueeze(2).shape)
        new_shape[2] = 2
        new_s = torch.zeros(new_shape)
        new_s[:,:,0] = 1 - s
        new_s[:,:,1] = s
        samples2.append(new_s)
    samples = samples2

    return means, samples, ConcatDataset(test_datasets)

import torch.distributions as td
from trustworthai.utils.losses_and_metrics.evidential_bayes_risks import *

def generate_means_and_samples_Evidential(splits=6, dataset_stride=2, temp=1, num_samples=10):
    print("strawberry")
    # load data
    clin_retriever = ClinScoreDataRetriever()
    
    models_folder = "/home/s2208943/ipdis/results/cross_validated_models/"
    #model_names = os.listdir(models_folder)

    means = []
    samples = []

    test_datasets = []

    for split in range(splits):
        # load specific data split
        train_ds_clin, val_ds_clin, test_ds_clin = clin_retriever.load_clinscores_data(
            combine_all=False,
            test_proportion=0.15, 
            validation_proportion=0.12,
            seed=3407,
            cross_validate=True,
            cv_split=split,
            cv_test_fold_smooth=1,
        )
        test_datasets.append(test_ds_clin)
        print("size: ", len(test_ds_clin))

        model_name = f"evidential_dice+xent_kl01_split{split}/"
        model_path = models_folder + model_name
        with open(model_path + "best_ckpt.txt") as f:
            lines = f.readlines()
            args_lines = [l[:-1].split(": ") for l in lines[1:]]
            args_lines = [f'--{l[0]} {l[1]}' for l in args_lines]
            args_line = " ".join(args_lines)
            parser = construct_parser()
            args = parser.parse_args(shlex.split(args_line))

        model_raw, loss = load_deterministic(args)
        print(args.dropout_p)

        # load the model
        model = load_best_checkpoint(model_raw, loss, model_path)

        with open(model_path + "best_ckpt.txt") as f:
            lines = f.readlines()
            args_lines = [l[:-1].split(": ") for l in lines[1:]]
            args_lines = [f'--{l[0]} {l[1]}' for l in args_lines]
            args_line = " ".join(args_lines)
            parser = construct_parser()
            args = parser.parse_args(shlex.split(args_line))

        model_raw, loss = load_deterministic(args)

        # load the model
        model = load_best_checkpoint(model_raw, loss, model_path)

        dataskip = dataset_stride
        for i, data in enumerate(tqdm(test_ds_clin, position=0, leave=True)):
            if i % dataskip == 0:
                x = data[0]
                with torch.no_grad():
                    model.eval()
                    logits = model(x.swapaxes(0,1).cuda()).cpu()
                    evidence = softplus_evidence(logits)
                    alpha = get_alpha(evidence)
                    # print(alpha.shape)
                    S = get_S(alpha)
                    K = alpha.shape[1]
                    mean_p_hat = get_mean_p_hat(alpha, S)
                    means.append(mean_p_hat)
                    dist = td.Dirichlet(alpha.moveaxis(1,-1))
                    ind_samples = []
                    samples.append(dist.sample((num_samples,)).moveaxis(-1, 2))

    return means, samples, ConcatDataset(test_datasets)

import matplotlib.pyplot as plt
from trustworthai.utils.uncertainty_maps.entropy_map import entropy_map_from_samples


def plot_example(save_dir, test_datasets, means, ent_maps, scan_index, slice_index=25, stride=2):
    plt.imshow(test_datasets[scan_index * stride][0][0][slice_index],
           cmap='gray',
           vmin=None, vmax=None, origin='lower')
    plt.axis('off')
    save(save_dir, f"flair_{scan_index}_{slice_index}", is_img=True)
    
    plt.imshow(test_datasets[scan_index * stride][1][0][slice_index],
           cmap='gray',
           vmin=None, vmax=None, origin='lower')
    plt.axis('off')
    save(save_dir, f"GT_{scan_index}_{slice_index}", is_img=True)
    
    plt.imshow(means[scan_index][slice_index].argmax(dim=0),
           cmap='gray',
           vmin=None, vmax=None, origin='lower')
    plt.axis('off')
    save(save_dir, f"mean_{scan_index}_{slice_index}", is_img=True)
    
    plt.imshow(ent_maps[scan_index][slice_index],
           cmap='magma',
           vmin=0, vmax=0.7, origin='lower')
    plt.axis('off')
    save(save_dir, f"umap_{scan_index}_{slice_index}", is_img=True)

def iou(a, b):
    intersection = (a==1) & (b==1)
    union = (a==1) | (b==1)
    return intersection.sum()/union.sum()

def reorder_samples(sample):
    slice_volumes = sample.sum(dim=(-1, -2))
    slice_volume_orders = torch.sort(slice_volumes.T, dim=1)[1]
    
    # rearrange the samples into one...
    new_sample = torch.zeros(sample.shape).to(sample.device)
    for i, slice_volumes_orders in enumerate(slice_volume_orders):
        for j, sample_index in enumerate(slice_volumes_orders):
            new_sample[j][i] = sample[sample_index][i]
            
    return new_sample

def iou_GED(means, ys3d_test, samples, reorder=False):
    geds = []
    
    for i in tqdm(range(len(means)), position=0, leave=True):
        y = ys3d_test[i].cuda()
        ss = samples[i].cuda().argmax(dim=2)
        
        if reorder:
            ss = reorder_samples(ss)
        
        dists_ab = 0
        
        # print(y.sum())
        
        for s in ss:
            pred = s#.argmax(dim=1)
            dists_ab += (1 - iou(pred, y).item())
            # print(dists_ab)
            # print(s.shape)
        
        dists_ab /= ss.shape[0]
        dists_ab *= 2
        
        dists_ss = 0
        for s1 in ss:
            for s2 in ss:
                dists_ss += (1 - iou(s1, s2).item())
        
        dists_ss /= (ss.shape[0] ** 2)
        
        ged = dists_ab - dists_ss
        if not np.isnan(ged):
            geds.append(ged)
            
        #break
        
    return torch.Tensor(geds)

def per_model_chal_stats(preds3d, ys3d):
    stats = []
    for i in tqdm(range(len(ys3d)), position=0, leave=True):
        ind_stats = do_challenge_metrics(ys3d[i].type(torch.long), preds3d[i].argmax(dim=1).type(torch.long))
        stats.append(ind_stats)

    tstats = torch.Tensor(stats)
    dices = tstats[:,0]
    hd95s = tstats[:,1]
    avds = tstats[:,2]
    recalls = tstats[:,3]
    f1s = tstats[:,4]

    data = {"dice":dices, "hd95":hd95s, "avd":avds, "recall":recalls, "f1":f1s}

    return data

def write_model_metric_results(results_file, data):
    for key in data.keys():
        print_and_write(results_file, key, newline=1)
        print_and_write(results_file, data[key])
        
        # ignore any bad images, that will cause values of 0 to appear
        values = data[key]
        ignores = (values < 0.001) | torch.isnan(values)
        if key == 'avd':
            ignores = ignores | (values == 100)
        values = values[~ignores]
        
        print_and_write(results_file, f"{key} mean", newline=1)
        print_and_write(results_file, values.mean())
        
        print_and_write(results_file, f"{key} standard error", newline=1)
        print_and_write(results_file, scipy.stats.sem(values))
        
        
def fast_dice(pred, target):
    p1 = (pred == 1)
    t1 = (target == 1)
    intersection = (pred == 1) & (target == 1)
    numerator = 2 * intersection.sum()
    denominator = p1.sum() + t1.sum()
    return (numerator/(denominator + 1e-30)).item()

def fast_avd(pred, target):
    p1 = pred.sum()
    t1 = target.sum()
    
    return ((p1 - t1).abs() / t1) * 100

def fast_vd(pred, target):
    p1 = pred.sum()
    t1 = target.sum()
    
    return ((p1 - t1) / t1) * 100


def fast_dice(pred, target):
    p1 = (pred == 1)
    t1 = (target == 1)
    intersection = (pred == 1) & (target == 1)
    numerator = 2 * intersection.sum()
    denominator = p1.sum() + t1.sum()
    return (numerator/(denominator + 1e-30)).item()

def fast_avd(pred, target):
    p1 = pred.sum()
    t1 = target.sum()
    
    return ((p1 - t1).abs() / t1) * 100

def fast_vd(pred, target):
    p1 = pred.sum()
    t1 = target.sum()
    
    return ((p1 - t1) / t1) * 100


def vd_per_sample(ys3d, samples3d, reorder=False):
    device='cuda'
    results = []
    for ind in tqdm(range(len(samples3d)), position=0, leave=True, ncols=150):
        sample_results = []
        sample_set = samples3d[ind].to(device).argmax(dim=2)
        if reorder:
            sample_set = reorder_samples(sample_set.cuda()).to(device)
            
        y = ys3d[ind].to(device)
        
        for y_hat in sample_set:
            sample_results.append(fast_vd(y_hat, y))

        results.append(sample_results)

    results = torch.stack([torch.Tensor(ds) for ds in results], dim=0)
    
    return results

def GT_volumes(ys3d):
    volumes = []
    for y in ys3d:
        volumes.append(y.sum())
    return torch.Tensor(volumes)

def sample_diversity_plot(save_folder, sample_metrics_3d, metric_name):
    # sort in order of quality
    order = torch.sort(torch.median(sample_metrics_3d, dim=1)[0])[1]
    plt.figure(figsize=(20, 5))
    plt.boxplot(sample_metrics_3d[order]);
    plt.ylim(-0.5, 2.5);
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.ylabel(f"{metric_name}")
    #plt.yscale('log')
    plt.xlabel("Individuals")
    save(save_folder, "sample_diversity_plot")
    
def sample_diversity_plot_by_volume(save_folder, sample_metrics_3d, volumes, metric_name):
    # sort in order of quality
    order = torch.sort(volumes)[1]
    fig, ax = plt.subplots(figsize=(20, 5))
    #plt.figure(figsize=(20, 5))
    ax.boxplot(sample_metrics_3d[order]);
    # ax.set_xticklabels(np.array(volumes[order] * 0.003))
    # plt.xticks(rotation = 90)
    plt.ylim(-0.5, 2.5);
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.ylabel(f"{metric_name}")
    #plt.yscale('log')
    plt.xlabel("Individuals, sorted by WMH Volume")
    
    save(save_folder, "sample_diversity_plot")
    
def filtered_dice(pred, ent_map, target, threshold):
    uncertain_locs = ent_map < threshold
    remaining_pred = pred[uncertain_locs]
    remaining_target = target[uncertain_locs]
    
    return fast_dice(remaining_pred, remaining_target)

def filtered_tps_score(pred, ent_map, target, threshold):
    total_tps = ((pred == 1) & (target == 1)).sum().item()
    uncertain_locs = ent_map < threshold
    
    filtered_tps = ((pred[uncertain_locs] == 1) & (target[uncertain_locs] == 1)).sum().item()
    
    return (total_tps - filtered_tps) / (total_tps + 1e-30)

def filtered_tns_score(mask, pred, ent_map, target, threshold):
    mask = mask.type(torch.bool)
    pred = pred[mask]
    ent_map = ent_map[mask]
    target = target[mask]
    
    total_tns = ((pred == 0) & (target == 0)).sum().item()
    uncertain_locs = ent_map < threshold
    
    filtered_tns = ((pred[uncertain_locs] == 0) & (target[uncertain_locs] == 0)).sum().item()
    
    return (total_tns - filtered_tns) / (total_tns + 1e-30)

# now compute the UEO, sUEO and sUEO score....
def sUEO(pred, ent_map, target):
    errors = (pred != target)
    
    numerator = 2 * (ent_map * errors).sum()
    denominator = (errors**2).sum() + (ent_map**2).sum()
    
    return (numerator / denominator).item()

def UEO_per_threshold_analysis(save_folder, text_results_file, uncertainty_thresholds, ys3d, ind_ent_maps, means, max_ent):
    ueos = [[] for _ in range(len(uncertainty_thresholds))]
                              
    for i in tqdm(range(len(ys3d)), position=0, leave=True):
        pred = means[i].argmax(dim=1).cuda()
        target = ys3d[i].cuda()
        ent = ind_ent_maps[i].cuda()
        
        if pred.sum() == 0:
            continue
        
        for j, t in enumerate((uncertainty_thresholds)):
            ueos[j].append(sUEO(pred, (ent > t).type(torch.float32), target))
    
    ueos = torch.stack([torch.Tensor(ind_ueo) for ind_ueo in ueos], dim=0)
    ueos = ueos.mean(dim=1)

    best_index = torch.Tensor(ueos).argmax()
    print_and_write(text_results_file, f"best tau for max UEO", newline=1)
    print_and_write(text_results_file, uncertainty_thresholds[best_index])
    print_and_write(text_results_file, "max UEO", newline=1)
    print_and_write(text_results_file, ueos[best_index])

    print_and_write(text_results_file, f"UEO per tau", newline=1)
    print_and_write(text_results_file, torch.Tensor(ueos))


    plt.plot(uncertainty_thresholds, ueos)
    plt.xlabel("τ")
    plt.ylabel("UEO")
    save(save_folder, "UEO")

    print_and_write(text_results_file, f"UEO tau AUC", newline=1)
    print_and_write(text_results_file, metrics.auc(uncertainty_thresholds/max_ent, ueos))
    
# we can compute the calibration using the mean confidence from our uncertainty maps - and just leave it at that. Nice.
# do mean conf, min conf, and max conf. Nice.

def calibration_over_samples(save_folder, results_file, means3d, samples3d, ys3d, do_normalize, mode="mean_conf"):
    bins = 10 + 1 # for the 0 bin
    bin_batch_accuracies = [[] for b in range(bins)]
    bin_batch_confidences = [[] for b in range(bins)]
    bin_batch_sizes = [[] for b in range(bins)]
    bin_counts = [0 for b in range(bins)]
    for batch_idx in tqdm(range(len(ys3d)), ncols=150, position=0, leave=True): # skip the last batch with a different shape
        batch_t = ys3d[batch_idx].squeeze().cuda()
        batch_samples = samples3d[batch_idx].cuda()

        if batch_t.shape[0] < 10:
            continue # skip last batch if it is very small.

        # get probabilities
        if do_normalize:
            probs = normalize_samples(batch_samples)
        else:
            probs = batch_samples
        p1s = probs[:,:,1]
        
        if mode == "all_samples":
            p1s = p1s # ie do nothing, use each sample
        elif mode == "mean_conf":
            p1s = p1s.mean(dim=0)
        elif mode == "min_conf":
            p1s = p1s.min(dim=0)[0]
        elif mode == "median_conf":
            p1s = p1s.median(dim=0)[0]
        elif mode == "max_conf":
            p1s = p1s.max(dim=0)[0]
        elif mode == "mean_only":
            if do_normalize:
                p1s = torch.softmax(means3d[batch_idx].cuda(), dim=1)[:,1]
            else:
                p1s = means3d[batch_idx][:,1]
        else:
            raise ValueError(f"mode: {mode} not accepted") 
            

        # split into bins
        bin_ids = place_in_bin(p1s)

        # compute counts
        for i in range(bins):
            is_in_bin = (bin_ids == (i / 10))
            # print(is_in_bin.shape)
            # print(batch_t.shape)

            # number of elements in each bin
            num_elem = torch.sum(is_in_bin).item()
            # if num_elem == 0:
            #     print("zero")

            # number of predictions = to class 1
            c1_acc = batch_t.expand(p1s.shape)[is_in_bin].sum() / num_elem

            # if torch.isnan(c1_acc):
            #     print("acc_nan")

            # average confidence of values in that bin
            c1_conf = p1s[is_in_bin].mean()

            # if torch.isnan(c1_conf):
            #     print("conf_nan")
                
            if torch.isnan(c1_conf) or torch.isnan(c1_acc) or num_elem == 0:
                #print("conf_nan") # just skip for this bin for this indivudal if they don't have have a prediction
                # with a confidence in this bin.
                continue

            bin_batch_accuracies[i].append(c1_acc.item())
            bin_batch_confidences[i].append(c1_conf.item())
            bin_batch_sizes[i].append(num_elem)

    bin_sizes = [torch.Tensor(bbs).sum() for bbs in bin_batch_sizes]
    bin_accuracies = [torch.Tensor([bin_batch_accuracies[i][j] * bin_batch_sizes[i][j] / bin_sizes[i] for j in range(len(bin_batch_accuracies[i]))]).sum().item() for i in range(len(bin_sizes))]
    bin_confidences = [torch.Tensor([bin_batch_confidences[i][j] * bin_batch_sizes[i][j] / bin_sizes[i] for j in range(len(bin_batch_confidences[i]))]).sum().item() for i in range(len(bin_sizes))]

    print_and_write(results_file, f"{mode} calibration curve data: ")

    print_and_write(results_file, f"{mode} bin_accuracies: ", newline=1)
    print_and_write(results_file, str(bin_accuracies))

    print_and_write(results_file, f"{mode} bin_confidences: ", newline=1)
    print_and_write(results_file, str(bin_confidences))

    total_size = torch.sum(torch.Tensor(bin_sizes)[1:])
    ece = torch.sum( (torch.Tensor(bin_sizes)[1:]/ total_size) * (torch.abs(torch.Tensor(bin_accuracies)[1:] - torch.Tensor(bin_confidences)[1:])))
    print_and_write(results_file, f"{mode} EXPECTED CALIBRATION ERROR", newline=1)
    print("note we skip the first bin due to its size")
    print_and_write(results_file, ece)

    plt.plot(bin_confidences, bin_accuracies)
    plt.plot([0,1],[0,1]);
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy");
    save(save_folder, f"{mode} calibration")
    
def metric_across_samples_analysis(results_text_file, ys3d, samples3d, reorder=False, metric="dice", print_all_results=False):
    if metric=="dice":
        func = fast_dice
        order_func = torch.max
        device="cuda"
    elif metric=="avd":
        func = fast_avd
        device='cuda'
        order_func = torch.min
    elif metric=="vd":
        func = fast_vd
        device='cuda'
        order_func = torch.min
    
    results = []
    for ind in tqdm(range(len(samples3d)), position=0, leave=True, ncols=150):
        sample_results = []
        sample_set = samples3d[ind].to(device).argmax(dim=2)
        if reorder:
            sample_set = reorder_samples(sample_set.cuda()).to(device)
            
        y = ys3d[ind].to(device)
        
        try:
            for y_hat in sample_set:
                sample_results.append(func(y_hat, y))

            results.append(sample_results)
        except ZeroDivisionError:
            pass # ignore the values that arise from a dodgy flair

    results = torch.stack([torch.Tensor(ds) for ds in results], dim=0)
    
    best = order_func(results, dim=1)[0]
    
    reordername = "reordered " if reorder else ""
    
    if print_all_results:
        print_and_write(results_text_file, reordername + f"per sample {metric}", newline=1)
        print_and_write(results_text_file, results)
    print_and_write(results_text_file, reordername + f"best {metric} mean", newline=1)
    print_and_write(results_text_file, best.mean())
    print_and_write(results_text_file, reordername + f"best {metric} st_err", newline=1)
    print_and_write(results_text_file, scipy.stats.sem(best))
    
    return results


################################### The actual run script ##################################
################################### Nice ###################################################


def run_eval(args):
    
    name = args.name
    splits = args.splits
    stride = args.stride
    temp = 1
    num_samples = 10
        
    
    name_to_method_map = {
        "softmax_ent":[generate_means_and_samples_CategoricalSoftmax],
        "mc_drop":[generate_means_and_samples_MC_Dropout],
        "evidential":[generate_means_and_samples_Evidential],
        "ensemble":[generate_means_and_samples_Ensemble],
        "SSN_ensemble":[generate_means_and_samples_SSN_Ens],
        "SSN_ensemble_mean":[generate_means_and_samples_SSN_Ens_Mean],
        "SSN":[generate_means_and_samples_SSN],
        "SSN_Ind":[generate_means_and_samples_SSN],
        "P-Unet":[generate_means_and_samples_PUNet],
    }
    
    method_func = name_to_method_map[name][0]
    if name == "SSN_Ind":
        means, samples, test_datasets = method_func(splits=splits, dataset_stride=stride, temp=temp, num_samples=num_samples, independent=True)
    elif name == "softmax_ent":
        means, samples, test_datasets = method_func(splits=splits, dataset_stride=stride, temp=temp, num_samples=num_samples, loss_name=args.determin_loss_name)
    else:
        means, samples, test_datasets = method_func(splits=splits, dataset_stride=stride, temp=temp, num_samples=num_samples)
        
    if "softmax" in name:
        name += "_" + args.determin_loss_name

    save_folder = f"/home/s2208943/ipdis/UQ_WMH_methods/trustworthai/run/results/{name}/"
    results_file = save_folder + "text_results.txt"

    try:
        os.mkdir(save_folder)
        os.mkdir(save_folder + "/images")
    except:
        pass

    with open(results_file, "w") as f:
        print_and_write(results_file, "begin results")
       
    
    # GET THE UMAPS
    # softmax entropy
    if "softmax" in name:
        ent_maps = [entropy_map_from_mean(means[scan_index], do_normalize=True) for scan_index in range(len(means))]

    # evidential
    elif "evidential" in name:
        ent_maps = [entropy_map_from_mean(means[scan_index], do_normalize=False) for scan_index in range(len(means))]

    else:
        ent_maps = [entropy_map_from_samples(samples[scan_index]) for scan_index in range(len(means))]
      
    
    # collect input data
    xs3d_test = []
    ys3d_test = []

    for i, data in enumerate(test_datasets):
        if i % stride == 0:
            ys3d_test.append(data[1].squeeze())
            xs3d_test.append(data[0])
            
    # do some plotting
    # plot_example(save_folder, test_datasets, means, ent_maps, 87, 15, stride)
    # plot_example(save_folder, test_datasets, means, ent_maps, 103, 35,stride)
    # plot_example(save_folder, test_datasets, means, ent_maps, 52, 35, stride)
    # plot_example(save_folder, test_datasets, means, ent_maps, 1, 25, stride)
    plot_example(save_folder, test_datasets, means, ent_maps, 0, 25, stride)
    # plot_example(save_folder, test_datasets, means, ent_maps, 57, 25, stride)
    # plot_example(save_folder, test_datasets, means, ent_maps, 60, 30, stride)
    # plot_example(save_folder, test_datasets, means, ent_maps, 70, 25, stride)
    
    plt.style.use('fivethirtyeight')
    
    # energy distance score
    geds_reordered = iou_GED(means, ys3d_test, samples, reorder=True)
    
    print_and_write(results_file, "reordered GED values", newline=1)
    print_and_write(results_file, geds_reordered)

    print_and_write(results_file, "reordered GED mean", newline=1)
    print_and_write(results_file, geds_reordered.mean())

    print_and_write(results_file, "reordered GED standard error", newline=1)
    print_and_write(results_file, scipy.stats.sem(geds_reordered))
    
    # sample diversity analysis
    standard_sample_dices = metric_across_samples_analysis(results_file, ys3d_test, samples, reorder=False, metric="dice")
    reorder_sample_dices = metric_across_samples_analysis(results_file, ys3d_test, samples, reorder=True, metric="dice")
    standard_sample_avds = metric_across_samples_analysis(results_file, ys3d_test, samples, reorder=False, metric="avd")
    reorder_sample_avds = metric_across_samples_analysis(results_file, ys3d_test, samples, reorder=True, metric="avd")
    
    # standard_sample_VDS = vd_per_sample(ys3d_test, samples, reorder=False)
    reorder_sample_VDS = vd_per_sample(ys3d_test, samples, reorder=True)
    gt_vols = GT_volumes(ys3d_test)
    sample_diversity_plot_by_volume(save_folder, reorder_sample_VDS*0.003, gt_vols, "Volume Difference (%)")
    
    # bras and sueo scores
    uncertainty_thresholds = torch.arange(0, 0.7, 0.01)

    filtered_dices = []
    filtered_tns = []
    filtered_tps = []

    for i in tqdm(range(len(means)), position=0, leave=True):
        mean = means[i].cuda().argmax(dim=1)
        ent_map = ent_maps[i].cuda()
        y = ys3d_test[i].cuda()
        mask = xs3d_test[i][1].cuda()

        ind_filtered_dices = []
        ind_filtered_tns = []
        ind_filtered_tps = []

        if mean.sum() == 0:
            continue

        for t in uncertainty_thresholds:
            ind_filtered_dices.append(filtered_dice(mean, ent_map, y, t))
            ind_filtered_tns.append(filtered_tns_score(mask, mean, ent_map, y, t))
            ind_filtered_tps.append(filtered_tps_score(mean, ent_map, y, t))

        filtered_dices.append(ind_filtered_dices)
        filtered_tns.append(ind_filtered_tns)
        filtered_tps.append(ind_filtered_tps)


    # we need to calculate the area under the curve of the dice, so that will be mean dice per threshold?
    filtered_dices = torch.stack([torch.Tensor(v) for v in filtered_dices], dim=0)
    filtered_tns = torch.stack([torch.Tensor(v) for v in filtered_tns], dim=0)
    filtered_tps = torch.stack([torch.Tensor(v) for v in filtered_tps], dim=0)
    
    fdice_curve = filtered_dices.mean(dim=0)
    tn_curve = filtered_tns.mean(dim=0)
    tp_curve = filtered_tps.mean(dim=0)
    max_ent = math.log(0.5)
    bras_score = (1/3) * (
        metrics.auc(uncertainty_thresholds/max_ent, fdice_curve)
         + (1 - metrics.auc(uncertainty_thresholds/max_ent, tn_curve))
         + (1 - metrics.auc(uncertainty_thresholds/max_ent, tp_curve))
    )
    print_and_write(results_file, "bras score", newline=1)
    print_and_write(results_file, bras_score)
    
    sUEOs = []
    for i in tqdm(range(len(means)), position=0, leave=True):
        pred = means[i].argmax(dim=1).cuda()
        target = ys3d_test[i].cuda()
        ent = ent_maps[i].cuda()

        if pred.sum() == 0:
            continue

        sUEOs.append(sUEO(pred, ent, target))

    sUEOs = torch.Tensor(sUEOs)
    print_and_write(results_file, "sUEO mean", newline=1)
    print_and_write(results_file, sUEOs.mean())

    print_and_write(results_file, "sUEO standard error", newline=1)
    print_and_write(results_file, scipy.stats.sem(sUEOs))
    
    UEO_per_threshold_analysis(save_folder, results_file, uncertainty_thresholds, ys3d_test, ent_maps, means, max_ent)
    
    if "softmax" in name:
        # for softmax entropy
        calibration_over_samples(save_folder, results_file, means, samples, ys3d_test, do_normalize=True, mode="mean_only")

    # evidential
    elif "evidential" in name:
        # For evidential
        calibration_over_samples(save_folder, results_file, means, samples, ys3d_test, do_normalize=False, mode="mean_only")

    else:
        # for other models
        calibration_over_samples(save_folder, results_file, means, samples, ys3d_test, do_normalize=True, mode="mean_conf")
        
        
    # collect the challenge data
    challenge_data = per_model_chal_stats(means, ys3d_test)
    write_model_metric_results(results_file, challenge_data)
    
    # the more expensive code goes at the end
    # 2D slice coverage
    uncertainty_thresholds = torch.arange(0, 0.7, 0.01)
    conn_comp_2d_analysis(save_folder, results_file, uncertainty_thresholds, ys3d=ys3d_test, means3d=means, ind_ent_maps=ent_maps)
    
    print("DONE!")

def script_parser():
    parser = argparse.ArgumentParser(description = "eval models")
    parser.add_argument('--name', type=str)
    parser.add_argument('--splits', default=6, type=int)
    parser.add_argument('--stride', default=2, type=int)
    parser.add_argument('--determin_loss_name', default='dice+xent', type=str)
    
    return parser    
    
if __name__ == '__main__':
    parser = script_parser()
    args = parser.parse_args()
    run_eval(args)