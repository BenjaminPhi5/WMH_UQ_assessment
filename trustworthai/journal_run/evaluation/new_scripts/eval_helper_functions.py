import torch
import pandas as pd
from trustworthai.analysis.evaluation_metrics.challenge_metrics import do_challenge_metrics
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from trustworthai.utils.fitting_and_inference.fitters.basic_lightning_fitter import StandardLitModelWrapper
from trustworthai.utils.fitting_and_inference.fitters.p_unet_fitter import PUNetLitModelWrapper

def load_best_checkpoint(model, loss, model_ckpt_folder, punet=False):
    # this is ultimately going to need to be passed a model wrapper when I implement P-Unet....
    
    # the path to the best checkpoint is stored as a single line in a txt file along with each model
    with open(os.path.join(model_ckpt_folder, "best_ckpt.txt"), "r") as f:
        ckpt_file = os.path.join(model_ckpt_folder, f.readlines()[0][:-1].split("/")[-1])
    
    try:
        if punet:
            return PUNetLitModelWrapper.load_from_checkpoint(ckpt_file, model=model, loss=loss, 
                                        logging_metric=lambda : None)
        return StandardLitModelWrapper.load_from_checkpoint(ckpt_file, model=model, loss=loss, 
                                        logging_metric=lambda : None)
    except:
        raise ValueError(f"ckpt {ckpt_file} couldn't be loaded, maybe it doesn't exist?")


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

def write_per_model_channel_stats(preds, ys3d_test, args, chal_results=None):
    if chal_results == None:
        chal_results = per_model_chal_stats(preds, ys3d_test)
    
    model_result_dir = os.path.join(args.repo_dir, args.result_dir, args.model_name)
    
    # save the results to pandas dataframes
    df = pd.DataFrame(chal_results)
    df['model_name'] = [args.model_name for _ in range(len(df))]
    
    df.to_csv(model_result_dir + "_individual_stats.csv")
    
    overall_stats = {"model_name":[args.model_name]}

    for key, value in chal_results.items():
        mean = value.mean()
        std = value.std(correction=1) # https://en.wikipedia.org/wiki/Bessel%27s_correction#Source_of_bias in this case we know the true mean..?
        # I want the standard deviation across this dataset, and I have a full sample, so I should use correction = 0? Or are we saying we have a limited sample of data from
        # an infinite distribution, and we want to know the model performance on that distribution, so correction = 1. Hmm this is a bit of a headache.
        conf_int = 1.96 * std / np.sqrt(len(value))

        lower_bound = mean - conf_int
        upper_bound = mean + conf_int

        overall_stats[f"{key}_mean"] = [mean.item()]
        overall_stats[f"{key}_95%conf"] = [conf_int.item()]
        
    overall_stats = pd.DataFrame(overall_stats)
    
    overall_stats.to_csv(model_result_dir + "_overall_stats.csv")
    
    
def per_sample_metric(samples, ys3d_test, f, do_argmax, do_softmax, minimise, take_abs=False):
    num_samples = len(samples[0])
    samples_f = []
    for i in tqdm(range(len(ys3d_test)), position=0, leave=True):
        scores = []
        y = ys3d_test[i].cuda()
        for j in range(num_samples):
            if do_argmax:
                s = samples[i][j].cuda().argmax(dim=1)
            elif do_softmax:
                s = torch.softmax(samples[i][j].cuda(), dim=1)
            else:
                s = samples[i][j].cuda()
            scores.append(f(s, y))
        scores = torch.Tensor(scores)
        samples_f.append(scores)

    samples_f = torch.stack(samples_f)

    if take_abs:
        samples_optimize = samples_f.abs()
    else:
        samples_optimize = samples_f
    
    if minimise:
        return samples_optimize.min(dim=1)[0], samples_f
    else:
        return samples_optimize.max(dim=1)[0], samples_f
    
    
def fast_rmse(pred, y, p=0.1):
    locs = (pred[:,1] > p)
    onehot = (y.unsqueeze(dim=1)==y.unique().view(1, -1, 1, 1)).type(torch.float32)
    
    pred = pred.moveaxis(1, -1)[locs]
    onehot = onehot.moveaxis(1, -1)[locs]
    
    return (pred - onehot).square().sum(dim=1).mean().sqrt()


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
    
    return (((p1 - t1).abs() / t1) * 100).item()

def fast_vd(pred, target):
    p1 = pred.sum()
    t1 = target.sum()
    
    return (((p1 - t1) / t1) * 100).item()


def get_xs_and_ys(eval_ds):
    xs3d_test = []
    ys3d_test = []

    for i, data in enumerate(eval_ds):
        ys3d_test.append(data[1].squeeze())
        xs3d_test.append(data[0])
        
    return xs3d_test, ys3d_test

def GT_volumes(ys3d):
    volumes = []
    for y in ys3d:
        volumes.append(y.sum())
    return torch.Tensor(volumes)