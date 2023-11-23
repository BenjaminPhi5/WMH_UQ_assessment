"""
the big boi evaluation script for basically every model. it all runs in one method so it can be called.
it defines a bunch of auxillary methods though within (sloppy copying) so that a bunch of information like
which model name it needs to read from etc and what its results file folder will be called etc go to.
"""

# I JUST COPIED ALL MY IMPORTS SO THAT I DO NOT MISS ANYTHING. MOST OF THESE ARE REDUNDANT.
print("strawberry")

import torch
import numpy as np
import torch.nn.functional as F

# dataset
from twaidata.torchdatasets.in_ram_ds import MRISegmentation2DDataset, MRISegmentation3DDataset
from torch.utils.data import DataLoader, random_split, ConcatDataset

# model
from trustworthai.models.uq_models.initial_variants.HyperMapp3r_deterministic import HyperMapp3r
from trustworthai.models.uq_models.initial_variants.HyperMapp3r_DDU import HyperMapp3rDDU
from trustworthai.models.uq_models.initial_variants.HyperMapp3r_SSN import HyperMapp3rSSN


# augmentation and pretrain processing
from trustworthai.utils.augmentation.standard_transforms import RandomFlip, GaussianBlur, GaussianNoise, \
                                                            RandomResizeCrop, RandomAffine, \
                                                            NormalizeImg, PairedCompose, LabelSelect, \
                                                            PairedCentreCrop, CropZDim
# loss function
from trustworthai.utils.losses_and_metrics.per_individual_losses import (
    log_cosh_dice_loss,
    TverskyLoss,
    FocalTverskyLoss,
    DiceLossMetric
)
from torch.nn import BCELoss, MSELoss, BCEWithLogitsLoss

# fitter
from trustworthai.utils.fitting_and_inference.fitters.basic_lightning_fitter import StandardLitModelWrapper
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl

# misc
import os
import torch
import matplotlib.pyplot as plt
import torch
from torchinfo import summary
import argparse

import torch.nn as nn
import torch
from torchmetrics import Metric
import math

import torch
import torch.nn as nn
from trustworthai.models.uq_models.drop_UNet import normalization_layer
import torch.nn.functional as F
from trustworthai.models.uq_models.initial_variants.HyperMapp3r_deterministic import HyperMapp3r
import torch.distributions as td
from typing import Tuple
from torch.utils.data import Dataset

import pyro
from torch.distributions.multivariate_normal import _batch_mahalanobis, _batch_mv
from torch.distributions.utils import _standard_normal, lazy_property
from pyro.distributions.torch_distribution import TorchDistribution

import math

import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from pyro.distributions.torch import Chi2
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as transforms

# ones i actually definately need in eval script
from tqdm import tqdm
import scipy.spatial
import seaborn as sns
from typing import Dict, Tuple
from sklearn import metrics

print("banana")

def place_in_bin(value):
    return torch.round(value, decimals=1)

def rolling_average(value, n, G):
    return value / n + ((n-1) / n) * G

def batch_rolling_average(values, n, G):
    """
    assumes all batches but the last batch are the same size
    """
    return values.sum() / (values.shape[0]*n) + ((n-1) / n) * G

def IOU(bs, cs, bs_are_targets=False, cs_are_targets=True):
    # for computing on a 3D image
    if not bs_are_targets:
        bs = bs.argmax(dim=1)
    if not cs_are_targets:
        cs = cs.argmax(dim=1)
    intersection = torch.sum(bs * cs)
    union = torch.logical_or(bs, cs).sum()
    return intersection / union

def all_samples_IOU(bs, cs):
    # for computing on a 3D image
    intersection = torch.sum(bs * cs, dim=1)
    union = torch.logical_or(bs, cs).sum(dim=1)
    return (intersection / union).mean()

def new_all_samples_IOU(bs, cs):
    ious = []
    for i in range(len(bs)):
        ious.append(IOU(bs, cs, True, True))
    return torch.Tensor(ious).mean()

def sample_diversity(samples):
    samples = samples.argmax(dim=2)
    ss = samples.shape[0]
    samples = samples.view(ss, -1)
    rolled = samples
    diversities = []
    for i in range(2):
        rolled = torch.roll(rolled, 1, 0)
        diversities.append(new_all_samples_IOU(samples, rolled))
    return 1. - torch.mean(torch.Tensor(diversities))

def sample_to_target_distance(samples, target):
    samples = samples.argmax(dim=2)
    target = target.unsqueeze(0).expand(samples.shape)
    ns = samples.shape[0]
    ind_distance = 1. - new_all_samples_IOU(samples.view(ns, -1),target.view(ns, -1))
    return ind_distance

def create_random_labels_map(classes: int) -> Dict[int, Tuple[int, int, int]]:
    labels_map: Dict[int, Tuple[int, int, int]] = {}
    for i in classes:
        labels_map[i] = torch.randint(0, 255, (3, ))
    labels_map[0] = torch.zeros(3)
    return labels_map

def labels_to_image(img_labels: torch.Tensor, labels_map: Dict[int, Tuple[int, int, int]]) -> torch.Tensor:
    """Function that given an image with labels ids and their pixels intrensity mapping, creates a RGB
    representation for visualisation purposes."""
    assert len(img_labels.shape) == 2, img_labels.shape
    H, W = img_labels.shape
    out = torch.empty(3, H, W, dtype=torch.uint8)
    for label_id, label_val in labels_map.items():
        mask = (img_labels == label_id)
        for i in range(3):
            out[i].masked_fill_(mask, label_val[i])
    return out

def show_components(img, labels):
    color_ids = torch.unique(labels)
    labels_map = create_random_labels_map(color_ids)
    labels_img = labels_to_image(labels, labels_map)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,12))

    # Showing Original Image
    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title("Orginal Image")

    #Showing Image after Component Labeling
    ax2.imshow(labels_img.permute(1,2,0).squeeze().numpy())
    ax2.axis('off')
    ax2.set_title("Component Labeling")
    
def sUEO(ent, target):
    numerator = 2 * torch.sum(ent * target)
    denominator = torch.sum((target**2) + (ent**2))
    return (numerator / denominator).item()

def run_evaluate(test_dataset, chall_dataset, gen_samples_func, model_name, root, do_softmax=True):
    EVALUATE(test_dataset, gen_samples_func, model_name, root, dataset_stride=1, do_softmax=do_softmax)
    EVALUATE(chall_dataset, gen_samples_func, model_name + "_DOMAIN_CHANGE", root, dataset_stride=2, do_softmax=do_softmax)
    
def save_wrapper(root, model_name, fname, small=False, img=False):
    plt.tight_layout()
    if small:
        plt.figure(figsize=(4,3))
    if img:    
        fname = "images/" + fname
    plt.savefig(root + "results/"+model_name + "/" + fname, bbox_inches = "tight")
    plt.show()
    plt.clf()
    plt.close()
    
def imsave_wrapper(root, model_name, fname, image, cmap, title="", small=False, img=False, vmin=None, vmax=None):
    plt.tight_layout()
    if small:
        plt.figure(figsize=(4,3))
    if img:    
        fname = "images/" + fname
    plt.axis('off')
    plt.title(title)
    plt.imsave(root + "results/"+model_name + "/" + fname + ".png", image, cmap=cmap, format="png", vmin=vmin, vmax=vmax)
    plt.imshow(image, cmap=cmap)
    plt.clf()
    plt.close()

def print_and_write_wrapper(root, model_name, message, clear_file=False, newline=2):
    mode = 'w' if clear_file else 'a'
    with open(root + "results/"+model_name + "/" + "text_results.txt", mode) as f:
        f.write(message)
        for _ in range(newline):
            f.write("\n")
    print(message)

def EVALUATE(test_dataset_3d, gen_samples_func, model_name, root, dataset_stride, do_softmax=True):
    import torch
    import numpy as np
    import torch.nn.functional as F

    # misc
    import os
    import matplotlib.pyplot as plt
    import torch.nn as nn
    from torchmetrics import Metric
    import math

    from typing import Tuple

    import math

    # ones i actually definately need in eval script
    from tqdm import tqdm
    import scipy.spatial
    import seaborn as sns
    from typing import Dict, Tuple
    from sklearn import metrics
    
    
    def entropy_map_from_samples(samples):
        "samples is of shape samples, batch size, channels, image dims  [s, b, c *<dims>]"
        if do_softmax:
            probs = torch.nn.functional.softmax(samples, dim=2)
        else:
            probs = samples
        pic = torch.mean(probs, dim=0)
        ent_map = torch.sum(-pic * torch.log(pic+1e-30), dim=1)

        return ent_map
    
    def VVC(v):
        if do_softmax:
            v = torch.nn.functional.softmax(v, dim=2)
        return torch.std(v) / torch.mean(v)

    # load data
    xs3d = []
    ys3d = []
    count = 0
    for x, y in test_dataset_3d:
        if (count % dataset_stride) == 0:
            xs3d.append(x)
            ys3d.append(y.squeeze())
        count += 1

    samples3d, means3d = gen_samples_func(xs3d, ys3d)

    # setup folder and files
    try:
        os.mkdir(root + "results/" + model_name)
        os.mkdir(root + "results/" + model_name + "/images")
    except:
        print("model folders must already exist")

    def save(fname, small=False, img=False):
        save_wrapper(root, model_name, fname, small=small, img=img)

    def imsave(fname, image, cmap, title="", small=False, img=False, vmin=None, vmax=None):
        imsave_wrapper(root, model_name, fname, image, cmap, title=title, small=small, img=img, vmin=vmin, vmax=vmax)

    def print_and_write(message, clear_file=False, newline=2):
        try:
            if ('torch' in message.type()) or ('Torch' in message.type):
                if len(message.shape) == 0:
                    message = str(message.numpy())
                else:
                    message = str(list(message.numpy()))
        except:
            pass
        print_and_write_wrapper(root, model_name, str(message), clear_file, newline)

    print_and_write("results for " + model_name, clear_file=True)

    scan_ids = [-1,-1,3,3,27,27,0,0]
    slice_ids = [38,26,32,18,33,37,22,24]

    for s in range(len(scan_ids)):

        scan_id = scan_ids[s]
        scan_ent_map = entropy_map_from_samples(samples3d[scan_id])

        slice_id = slice_ids[s]
        count = 0
        samples = samples3d[scan_id][:,slice_id]
        plt.figure(figsize=(30,8))
        for i in range(2*10):
            plt.subplot(2, 10, count+1)
            plt.imshow(samples[i].argmax(dim=0), cmap='gray')
            plt.title(i)
            plt.axis('off')
            count += 1
        save(f"all_samples: {scan_id}:{slice_id}", img=True)

        for i in range(len(samples)):
            # plt.imshow(samples[i].argmax(dim=0), cmap='gray')
            imsave(f"sample: {scan_id}:{slice_id}:{i}",
                   samples[i].argmax(dim=0),
                   cmap='gray',
                   title=i, img=True, small=True)

        # show ground truth and uncertainty map
        slice_ent_map = scan_ent_map[slice_id]
        imsave(f"GT: {scan_id}:{slice_id}:{i}", ys3d[scan_id][slice_id], cmap='gray', title=i, img=True, small=True)
        imsave(f"ent_map: {scan_id}:{slice_id}", slice_ent_map, cmap=None, title=i, img=True, small=True, vmin=0, vmax=0.7)

    from tqdm import tqdm
    import difflib
    import numpy as np
    import os
    import SimpleITK as sitk
    import scipy.spatial
    import seaborn as sns
    import pandas as pd

    def getDSC(testImage, resultImage):    
        """Compute the Dice Similarity Coefficient."""
        # testArray   = sitk.GetArrayFromImage(testImage).flatten()
        # resultArray = sitk.GetArrayFromImage(resultImage).flatten()
        testArray = testImage.view(-1).cpu().numpy()
        resultArray = resultImage.view(-1).cpu().numpy()

        # similarity = 1.0 - dissimilarity
        return 1.0 - scipy.spatial.distance.dice(testArray, resultArray) 


    def getHausdorff(testImage, resultImage):
        """Compute the Hausdorff distance."""

        testImage = sitk.GetImageFromArray(testImage)
        resultImage = sitk.GetImageFromArray(resultImage)

        # Hausdorff distance is only defined when something is detected
        resultStatistics = sitk.StatisticsImageFilter()
        resultStatistics.Execute(resultImage)
        if resultStatistics.GetSum() == 0:
            return float('nan')

        # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
        eTestImage   = sitk.BinaryErode(testImage, (1,1,0) )
        eResultImage = sitk.BinaryErode(resultImage, (1,1,0) )

        hTestImage   = sitk.Subtract(testImage, eTestImage)
        hResultImage = sitk.Subtract(resultImage, eResultImage)    

        hTestArray   = sitk.GetArrayFromImage(hTestImage)
        hResultArray = sitk.GetArrayFromImage(hResultImage)   

        # Convert voxel location to world coordinates. Use the coordinate system of the test image
        # np.nonzero   = elements of the boundary in numpy order (zyx)
        # np.flipud    = elements in xyz order
        # np.transpose = create tuples (x,y,z)
        # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
        testCoordinates   = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hTestArray) ))]
        resultCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hResultArray) ))]


        # Use a kd-tree for fast spatial search
        def getDistancesFromAtoB(a, b):    
            kdTree = scipy.spatial.KDTree(a, leafsize=100)
            return kdTree.query(b, k=1, eps=0, p=2)[0]

        # Compute distances from test to result; and result to test
        dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
        dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)    

        return max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))


    def getLesionDetection(testImage, resultImage):    
        """Lesion detection metrics, both recall and F1."""
        testImage = sitk.GetImageFromArray(testImage)
        resultImage = sitk.GetImageFromArray(resultImage)

        # Connected components will give the background label 0, so subtract 1 from all results
        ccFilter = sitk.ConnectedComponentImageFilter()    
        ccFilter.SetFullyConnected(True)

        # Connected components on the test image, to determine the number of true WMH.
        # And to get the overlap between detected voxels and true WMH
        ccTest = ccFilter.Execute(testImage)    
        lResult = sitk.Multiply(ccTest, sitk.Cast(resultImage, sitk.sitkUInt32))

        ccTestArray = sitk.GetArrayFromImage(ccTest)
        lResultArray = sitk.GetArrayFromImage(lResult)

        # recall = (number of detected WMH) / (number of true WMH) 
        nWMH = len(np.unique(ccTestArray)) - 1
        if nWMH == 0:
            recall = 1.0
        else:
            recall = float(len(np.unique(lResultArray)) - 1) / nWMH

        # Connected components of results, to determine number of detected lesions
        ccResult = ccFilter.Execute(resultImage)
        lTest = sitk.Multiply(ccResult, sitk.Cast(testImage, sitk.sitkUInt32))

        ccResultArray = sitk.GetArrayFromImage(ccResult)
        lTestArray = sitk.GetArrayFromImage(lTest)

        # precision = (number of detections that intersect with WMH) / (number of all detections)
        nDetections = len(np.unique(ccResultArray)) - 1
        if nDetections == 0:
            precision = 1.0
        else:
            precision = float(len(np.unique(lTestArray)) - 1) / nDetections

        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * (precision * recall) / (precision + recall)

        return recall, f1    


    def getAVD(testImage, resultImage):   
        """Volume statistics."""
        testImage = sitk.GetImageFromArray(testImage)
        resultImage = sitk.GetImageFromArray(resultImage)

        # Compute statistics of both images
        testStatistics   = sitk.StatisticsImageFilter()
        resultStatistics = sitk.StatisticsImageFilter()

        testStatistics.Execute(testImage)
        resultStatistics.Execute(resultImage)

        return float(abs(testStatistics.GetSum() - resultStatistics.GetSum())) / float(testStatistics.GetSum()) * 100

    def do_challenge_metrics(testImage, resultImage):
        """Main function"""
        dsc = getDSC(testImage, resultImage)
        try:
            h95 = getHausdorff(testImage, resultImage)
        except:
            h95 = 100
        avd = getAVD(testImage, resultImage)    
        recall, f1 = getLesionDetection(testImage, resultImage)    

        # print('Dice',                dsc,       '(higher is better, max=1)')
        # print('HD',                  h95, 'mm',  '(lower is better, min=0)')
        # print('AVD',                 avd,  '%',  '(lower is better, min=0)')
        # print('Lesion detection', recall,       '(higher is better, max=1)')
        # print('Lesion F1',            f1,       '(higher is better, max=1)')

        return dsc, h95, avd, recall, f1

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

    data = per_model_chal_stats(means3d, ys3d)

    def write_model_metric_results(data):
        print_and_write(f"dice", newline=1)
        print_and_write(data["dice"],newline=1)
        print_and_write(f"hd95",newline=1)
        print_and_write(data["hd95"],newline=1)
        print_and_write(f"avd",newline=1)
        print_and_write(data["avd"],newline=1)
        print_and_write(f"recall",newline=1)
        print_and_write(data["recall"],newline=1)
        print_and_write(f"f1",newline=1)
        print_and_write(data["f1"])

    write_model_metric_results(data)

    ### HOW DO AVERAGE DICE AND BEST DICE IMPROVE WITH SAMPLES
    def dice(y_pred, y_true):
        if do_softmax:
            y_pred = torch.nn.functional.softmax(y_pred, dim=1).argmax(dim=1)
        else:
            y_pred = y_pred.argmax(dim=1)
        #print(y_pred.shape, y_true.shape)
        denominator = torch.sum(y_pred) + torch.sum(y_true)
        numerator = 2. * torch.sum(torch.logical_and(y_pred, y_true))
        return numerator / denominator

    def AVD(y_pred, y_true):
        y_pred = y_pred.argmax(dim=1)
        avd = (y_pred.sum() - y_true.sum()).abs() / y_true.sum() * 100
        return avd.item()

    avds_mean = []
    for ind in range(len(ys3d)):
        mu = means3d[ind]
        target = ys3d[ind]
        avds_mean.append(AVD(mu, target))

    print_and_write(f"mean AVD", newline=1)
    print_and_write(torch.Tensor(avds_mean).mean())

    # compute dice for the mean produced by the model
    dices_mean = []
    for ind in range(len(ys3d)):
        mu = means3d[ind]
        target = ys3d[ind]
        dices_mean.append(dice(mu, target))

    dices_mean = torch.Tensor(dices_mean)
    print_and_write(f"mean dice", newline=1)
    print_and_write(dices_mean.mean())

    # compute the dice per sample, per individual
    dices3d = []
    for ind in tqdm(range(len(samples3d)), position=0, leave=True, ncols=150):
        sample_dices = []
        for s in range(len(samples3d[ind])):
            y_hat = samples3d[ind][s]
            y = ys3d[ind]
            sample_dices.append(dice(y_hat, y))
        dices3d.append(sample_dices)

    tensor_alldice3d = torch.stack([torch.Tensor(ds) for ds in dices3d], dim=0).swapaxes(0,1)

    # best dice mean. This is a little dissapointing.
    bdm = tensor_alldice3d.max(dim=0)[0].mean()
    print_and_write(f"best_dice_mean", newline=1)
    print_and_write(bdm)

    plt.style.use('fivethirtyeight')

    # sort in order of quality
    order = torch.sort(torch.median(tensor_alldice3d, dim=0)[0])[1]
    plt.figure(figsize=(20, 5))
    plt.boxplot(tensor_alldice3d.T[order]);
    plt.ylim(0, 1);
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.ylabel("Dice")
    plt.xlabel("Individuals")
    save("sample_diversity_plot")

    ### CALIBRATION


    print_and_write(f"check: len samples", newline=1)
    print_and_write(len(samples3d))

    # assess bin counts of p y = 1
    bins = 10 + 1 # for the 0 bin
    bin_batch_accuracies = [[] for b in range(bins)]
    bin_batch_confidences = [[] for b in range(bins)]
    bin_batch_sizes = [[] for b in range(bins)]
    bin_counts = [0 for b in range(bins)]
    for batch_idx in tqdm(range(len(ys3d)), ncols=150, position=0, leave=True): # skip the last batch with a different shape
        batch_t = ys3d[batch_idx].squeeze()
        batch_samples = samples3d[batch_idx]

        if batch_t.shape[0] < 10:
            continue # skip last batch if it is very small.

        # get probabilities
        if do_softmax:
            probs = torch.nn.functional.softmax(batch_samples, dim=2)
        else:
            probs = batch_samples
        p1s = probs[:,:,1]

        # split into bins
        bin_ids = place_in_bin(p1s)

        # compute counts
        for i in range(bins):
            is_in_bin = (bin_ids == (i / 10))
            # print(is_in_bin.shape)
            # print(batch_t.shape)

            # number of elements in each bin
            num_elem = torch.sum(is_in_bin).item()
            if num_elem == 0:
                print("zero")

            # number of predictions = to class 1
            c1_acc = batch_t.expand(p1s.shape)[is_in_bin].sum() / num_elem

            if torch.isnan(c1_acc):
                print("acc_nan")

            # average confidence of values in that bin
            c1_conf = p1s[is_in_bin].mean()

            if torch.isnan(c1_conf):
                print("conf_nan")

            bin_batch_accuracies[i].append(c1_acc)
            bin_batch_confidences[i].append(c1_conf)
            bin_batch_sizes[i].append(num_elem)

    bin_sizes = [torch.Tensor(bbs).sum() for bbs in bin_batch_sizes]
    bin_accuracies = [torch.Tensor([bin_batch_accuracies[i][j] * bin_batch_sizes[i][j] / bin_sizes[i] for j in range(len(bin_batch_accuracies[i]))]).sum().item() for i in range(len(bin_sizes))]
    bin_confidences = [torch.Tensor([bin_batch_confidences[i][j] * bin_batch_sizes[i][j] / bin_sizes[i] for j in range(len(bin_batch_confidences[i]))]).sum().item() for i in range(len(bin_sizes))]

    print_and_write("calibration curve data: ")

    print_and_write("bin_accuracies: ", newline=1)
    print_and_write(str(bin_accuracies))

    print_and_write("bin_confidences: ", newline=1)
    print_and_write(str(bin_confidences))

    total_size = torch.sum(torch.Tensor(bin_sizes)[1:])
    ece = torch.sum( (torch.Tensor(bin_sizes)[1:]/ total_size) * (torch.abs(torch.Tensor(bin_accuracies)[1:] - torch.Tensor(bin_confidences)[1:])))
    print_and_write(f"EXPECTED CALIBRATION ERROR", newline=1)
    print("note we skip the first bin due to its size")
    print_and_write(ece)

    plt.plot(bin_confidences, bin_accuracies)
    plt.plot([0,1],[0,1]);
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy");
    save("calibration")

    # SAMPLE DIVERSITY AND GENERALIZED ENERGY DISTANCE

    per_ind_diversities = []
    for s in tqdm(samples3d[::2], position=0, leave=True):
        per_ind_diversities.append(sample_diversity(s))

    overall_sample_diversity = torch.Tensor(per_ind_diversities).mean()
    print_and_write("sample diversity", newline=1)
    print_and_write(overall_sample_diversity)

    all_imgs_stot_distance = []
    for i in tqdm(range(len(ys3d)), position=0, leave=True):
        all_imgs_stot_distance.append(sample_to_target_distance(samples3d[i], ys3d[i]))

    print_and_write(f"generalized energy distance:", newline=1)
    print_and_write(2 * torch.Tensor(all_imgs_stot_distance).mean() - overall_sample_diversity)
    #2 * torch.Tensor(all_imgs_stot_distance).mean() - overall_sample_diversity

    # COMPUTE PAVPU METRICS
    # I am going to do it per patch, but take the average accuracy per patch (perhaps I should qc average dice as well, best dice, worst dice.
    uncetainty_thresholds = torch.arange(0, 0.7, 0.01)
    accuracy_threshold = 0.9
    window = 16
    stride = 16
    n_acs = [[] for i in range(len(uncetainty_thresholds))]
    n_aus = [[] for i in range(len(uncetainty_thresholds))]
    n_ics = [[] for i in range(len(uncetainty_thresholds))]
    n_ius = [[] for i in range(len(uncetainty_thresholds))]

    for batch_idx in tqdm(range(len(ys3d)), ncols=150, position=0, leave=True): # skip the last batch with a different shape
        batch_t = ys3d[batch_idx].squeeze()
        batch_samples = samples3d[batch_idx]
        batch_mean = means3d[batch_idx]
        ent = entropy_map_from_samples(batch_samples)

        # get probabilities
        if do_softmax:
            probs = torch.nn.functional.softmax(batch_samples, dim=2)
        else:
            probs = batch_samples
        pred_classes = probs.argmax(dim=2)
        confidences = probs.max(dim=2)[0]

        # get average accuracy of each sample using the mean
        # or I could treat each patch of each sample as a separate thing but that is not what I am doing here.
        #avg_accuracy = ((batch_t.expand(pred_classes.shape) == pred_classes) * 1.).mean(dim=0)
        accuracy = batch_t == batch_mean.argmax(dim=1)

        # unroll predictions and targets and entropy
        t_unrolled = batch_t.unfold(-2, window, stride).unfold(-1, window, stride).reshape(-1, window, window)
        #accuracy_unrolled = avg_accuracy.unfold(-2, window, stride).unfold(-1, window, stride).reshape(-1, window, window)
        accuracy_unrolled = accuracy.unfold(-2, window, stride).unfold(-1, window, stride).reshape(-1, window, window)
        ent_unrolled = ent.unfold(-2, window, stride).unfold(-1, window, stride).reshape(-1, window, window)

        #accurate_patches = accuracy_unrolled > 0.9
        accurate_patches = accuracy_unrolled.type(torch.float32).mean(dim=(1,2)) > 0.9
        # print(accurate_patches.shape)

        # try applying around patches that have lesion burden.
        has_lesion = t_unrolled.mean(dim=(1,2)) > 0.

        # for each uncertainty threshold, compute the 4 numbers
        for i, uncert_t in enumerate(uncetainty_thresholds):
            #uncertain_patches = ent_unrolled > uncert_t
            uncertain_patches = ent_unrolled.mean(dim=(1,2)) > uncert_t
            # print(uncertain_patches.shape)

            n_acs[i].append(torch.sum(torch.logical_and(has_lesion, torch.logical_and(accurate_patches, ~uncertain_patches))))
            n_aus[i].append(torch.sum(torch.logical_and(has_lesion, torch.logical_and(accurate_patches, uncertain_patches))))
            n_ics[i].append(torch.sum(torch.logical_and(has_lesion, torch.logical_and(~accurate_patches, ~uncertain_patches))))
            n_ius[i].append(torch.sum(torch.logical_and(has_lesion, torch.logical_and(~accurate_patches, uncertain_patches))))

    n_acs_t = [torch.Tensor(n_acs[i]).sum() for i in range(len(uncetainty_thresholds))]
    n_aus_t = [torch.Tensor(n_aus[i]).sum() for i in range(len(uncetainty_thresholds))]
    n_ics_t = [torch.Tensor(n_ics[i]).sum() for i in range(len(uncetainty_thresholds))]
    n_ius_t = [torch.Tensor(n_ius[i]).sum() for i in range(len(uncetainty_thresholds))]

    p_acs = [n_acs_t[i] / (n_acs_t[i] + n_ics_t[i]) for i in range(len(uncetainty_thresholds))]
    p_aus = [n_ius_t[i] / (n_ius_t[i] + n_ics_t[i]) for i in range(len(uncetainty_thresholds))]
    pavpu = [(n_acs_t[i] + n_ius_t[i]) / (n_ius_t[i] + n_ics_t[i] + n_aus_t[i] + n_acs_t[i]) for i in range(len(uncetainty_thresholds))]

    plt.figure(figsize=(13,3))
    plt.subplot(1,3,1)
    plt.plot(uncetainty_thresholds, p_acs, c='g')
    plt.xlim((-0.01,0.7)); plt.ylim((-0.05,1.05))
    plt.xlabel("τ")
    plt.ylabel("p(acc|cert)")
    plt.subplot(1,3,2)
    plt.plot(uncetainty_thresholds, p_aus, c='g')
    plt.xlim((-0.01,0.7)); plt.ylim((-0.05,1.05))
    plt.ylabel("p(uncert|inacc)")
    plt.xlabel("τ")
    plt.subplot(1,3,3)
    plt.plot(uncetainty_thresholds, pavpu, c='g')
    plt.xlim((-0.01,0.7)); plt.ylim((-0.05,1.05))
    plt.ylabel("PAVPU")
    plt.xlabel("τ")
    save("pavpu")

    print_and_write(f"p_acs:", newline=1)
    print_and_write(torch.stack(p_acs))
    print_and_write(f"p_aus:", newline=1)
    print_and_write(torch.stack(p_aus))
    print_and_write(f"pavpu:", newline=1)
    print_and_write(torch.stack(pavpu))

    i=10
    #print_and_write(f"for tau at 01, tau pac, pau, pacpu: {uncetainty_thresholds[i]}, {p_acs[i]}, {p_aus[i]}, {pavpu[i]}")

    ent_thresh = 0.05

    # QUALITY CONTROL IN 3D - vcc corr coeff
    # generate entropy maps per individual
    ind_ent_maps = [entropy_map_from_samples(samples3d[i]) for i in range(len(ys3d))]


    vvcs = [VVC(samples3d[i]) for i in range(len(ys3d))]

    medians = torch.median(tensor_alldice3d, dim=0)[0]

    print_and_write("vvc correlation coefficient:", newline=1)
    print_and_write(torch.corrcoef(torch.stack([torch.Tensor(vvcs), medians]))[0][1])


    # TP TN FN DISTRIBUTION!!
    all_tps = []
    #all_tns = []
    all_fps = []
    all_fns = []

    with torch.no_grad():
        for i in tqdm(range(len(ys3d)), position=0, leave=True, ncols=150):
            samples = samples3d[i]
            mean = means3d[i]
            ent = ind_ent_maps[i].view(-1)
            mean_class = mean.argmax(dim=1).view(-1)
            y = ys3d[i]
            x = xs3d[i].swapaxes(0,1)
            y_flat = y.view(-1)

            tp_loc = torch.where(torch.logical_and(y_flat == 1, mean_class == 1))[0]
            #tn_loc = torch.where(torch.logical_and(torch.logical_and(y_flat == 0, mean_class == 0), x[:,1].reshape(-1) == 1))[0]
            fp_loc = torch.where(torch.logical_and(y_flat == 0, mean_class == 1))[0]
            fn_loc = torch.where(torch.logical_and(torch.logical_and(y_flat == 1, mean_class == 0), x[:,1].reshape(-1) == 1))[0]
            # print(tp_loc.shape)
            # print(ent.view(-1).shape)

            all_tps.append(ent[tp_loc])
            #all_tns.append(ent[tn_loc])
            all_fps.append(ent[fp_loc])
            all_fns.append(ent[fn_loc])

    tps = torch.cat(all_tps)
    #tns = torch.cat(all_tns)
    fps = torch.cat(all_fps)
    fns = torch.cat(all_fns)

    print_and_write(f"tp, fp, fn totals")
    print_and_write(str([tps.shape[0], fps.shape[0], fns.shape[0]]))

    print_and_write("TP mean", newline=1)
    print_and_write(tps.mean())
    print_and_write("FP mean", newline=1)
    print_and_write(fps.mean())
    print_and_write("FN mean", newline=1)
    print_and_write(fns.mean())

    print_and_write("TP ent values", newline=1)
    print_and_write(tps)
    print_and_write("FP ent values", newline=1)
    print_and_write(fps)
    print_and_write("FN ent values", newline=1)
    print_and_write(fns)


    plt.hist(tps, bins=100, color='r');
    plt.ylabel("Voxels per Bin")
    #plt.ylim((0, 350000))
    plt.xlabel("$H$")
    save("tps")

    plt.hist(fps, bins=100, color='r');
    plt.ylabel("Voxels per Bin")
    #plt.ylim((0, 50000))
    plt.xlabel("$H$")
    save("fps")

    plt.hist(fns, bins=100, color='r');
    plt.ylabel("Voxels per Bin")
    #plt.ylim((0, 155000))
    plt.xlabel("$H$")
    save("fns")

    j = -1
    ntps = len(tps)
    nfns = len(fns)
    nfps = len(fps)
    data = {"label":["TP" for _ in range(ntps)][0:j] + ["FN" for _ in range(nfns)][0:j] + ["FP" for _ in range(nfps)][0:j], "ent": torch.cat([tps[0:j], fns[0:j], fps[0:j]]).numpy()}

    plt.figure(figsize=(4, 2.5))
    sns.violinplot(x="label", y="ent", data=data, linewidth=0.5, inner=None)
    plt.ylim((-0.1, 0.8))
    plt.ylabel("$H$")
    save("types_violin")

    ### MISSING LESIONS IN 2D SLICES
    from typing import Dict, Tuple

    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    import numpy as np
    import kornia as K



    conncomp_outs = []

    for y in tqdm(ys3d, position=0, leave=True, ncols=150):
        labels_out = K.contrib.connected_components(y.unsqueeze(1).type(torch.float32), num_iterations=150)
        conncomp_outs.append(labels_out)

    # this is the 1 pixel is covered by the entropy
    #c_thresholds = [0.05, 0.1, 0.2, 0.3, 0.6]
    c_thresholds = [t.item() for t in uncetainty_thresholds]
    coverages = [0.1, 0.5, 0.9]
    missing_lesion_size_ent = []
    existing_lesion_size_ent = []

    missing_lesion_size_mean = []

    num_entirely_missed_lesions = {ct:0 for ct in c_thresholds}
    entirely_missed_lesions_size = {ct:[] for ct in c_thresholds}
    proportion_missing_lesion_covered_ent = {ct:[] for ct in c_thresholds}
    num_lesions = 0
    sizes = []
    missing_area_sizes = []
    missing_area_coverage = {ct:[] for ct in c_thresholds}
    for batch in tqdm(range(len(ys3d)), position=0, leave=True, ncols=150):
        for i in range(0, ys3d[batch].shape[0], 3):
            conncomps = conncomp_outs[batch][i]
            ent = ind_ent_maps[batch][i]
            mean = means3d[batch].argmax(dim=1)[i]

            ids = conncomps.unique()[1:]
            for idx in ids:
                cc = (conncomps == idx)
                num_lesions += 1
                size = torch.sum(cc)
                sizes.append(size)

                missing_area = (mean == 0) & cc
                ma_size = missing_area.sum()
                missing_area_sizes.append(ma_size)

                # get uncertain pixels for each threshold
                for tau in c_thresholds:
                    uncert = (ent > tau).type(torch.long)

                    # coverage proportion
                    coverage = (uncert * missing_area).sum() / ma_size
                    missing_area_coverage[tau].append(coverage)


                    if torch.max(mean * cc) == 0:
                        # proportion of those lesions that are missing from mean covered by uncertainty
                        proportion_missing_lesion_covered_ent[tau].append(torch.sum(uncert * cc) / size)

                        # lesions entirely missed by both mean prediction and uncertainty map
                        # i.e not a single voxel is identified as uncertain or mean, total silent failure.
                        if torch.max(uncert * cc) == 0:
                            num_entirely_missed_lesions[tau] += 1
                            entirely_missed_lesions_size[tau].append(size)

    # replace nans and convert to tensor
    for tau in c_thresholds:
        missing_area_coverage[tau] = torch.Tensor([c.item() if not torch.isnan(c) else 0 for c in missing_area_coverage[tau]])

    print_and_write("uncertainty thresholds", newline=1)
    print_and_write(uncetainty_thresholds)

    print_and_write("mean coverage of areas missed by mean as tau increases", newline=1)
    # for tau in c_thresholds:
    #     print_and_write(f"{tau}: {missing_area_coverage[tau].mean().item()}", newline=1)
    # print_and_write("",newline=1)
    print_and_write(torch.Tensor([missing_area_coverage[tau].mean().item() for tau in c_thresholds]))

    print_and_write("mean size of entirely missed lesions", newline=1)
    # for tau in c_thresholds:
    #     print_and_write(f"{tau}: {torch.Tensor(entirely_missed_lesions_size[tau]).mean().item()}", newline=1)
    # print_and_write("",newline=1)
    print_and_write(torch.Tensor([torch.Tensor(entirely_missed_lesions_size[tau]).mean().item() for tau in c_thresholds]))


    print_and_write("mean coverage of lesions entirely missed by the mean segmentation", newline=1)
    # for tau in c_thresholds:
    #     print_and_write(f"{tau}: {torch.Tensor(proportion_missing_lesion_covered_ent[tau]).mean().item()}", newline=1)
    # print_and_write("",newline=1)
    print_and_write(torch.Tensor([torch.Tensor(proportion_missing_lesion_covered_ent[tau]).mean().item() for tau in c_thresholds]))

    print_and_write("total number of missing lesions", newline=1)
    # for tau in c_thresholds:
    #     print_and_write(f"{tau}: {num_entirely_missed_lesions[tau]}", newline=1)
    # print_and_write("",newline=1)
    print_and_write(torch.Tensor([num_entirely_missed_lesions[tau] for tau in c_thresholds]))



    print_and_write("proportion of lesions entirely missed", newline=1)
    # for tau in c_thresholds:
    #     print_and_write(f"{tau}: {num_entirely_missed_lesions[tau]/num_lesions}", newline=1)
    # print_and_write("",newline=1)
    print_and_write(torch.Tensor([num_entirely_missed_lesions[tau]/num_lesions for tau in c_thresholds]))



    # print("total number of missing lesions")
    # print([(tau, num) for tau, num in num_entirely_missed_lesions.items()])
    # print("proportion of lesions entirely missed")
    # print([(tau, num/num_lesions) for tau, num in num_entirely_missed_lesions.items()])

    print_and_write(f"num lesions:", newline=1)
    print_and_write(num_lesions)


    ### EXAMINATION OF SHAPE BASED METRICS

    sueo_s = []
    for i in range(len(ys3d)):
        sueo_s.append(sUEO(ind_ent_maps[i], ys3d[i]))

    print_and_write(f"sUEO", newline=1)
    print_and_write(torch.mean(torch.Tensor(sueo_s)).item())

    # UEO = sUEO but U is thresholded binary now. We can plot it over tau, increasing in 0.05 steps
    ueos = []
    for t in tqdm(uncetainty_thresholds, position=0, leave=True):
        t_ueos = []
        for i in range(len(ys3d)):
            t_ueos.append((sUEO((ind_ent_maps[i] > t).type(torch.float32), ys3d[i])))
        ueos.append(torch.Tensor(t_ueos).mean().item())

    best_index = torch.Tensor(ueos).argmax()
    print_and_write(f"best tau for max UEO", newline=1)
    print_and_write(uncetainty_thresholds[best_index])
    print_and_write("max UEO", newline=1)
    print_and_write(ueos[best_index])

    print_and_write(f"UEO per tau", newline=1)
    print_and_write(torch.Tensor(ueos))


    plt.plot(uncetainty_thresholds, ueos)
    plt.xlabel("τ")
    plt.ylabel("UEO")
    save("UEO")

    print_and_write(f"tau AUC", newline=1)
    print_and_write(metrics.auc(uncetainty_thresholds, ueos))