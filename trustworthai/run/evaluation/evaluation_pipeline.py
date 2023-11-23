print("strawberry")

import os
import matplotlib.pyplot as plt
import torch

from trustworthai.analysis.sample_diversity.sample_distances import sample_diversity, sample_to_target_distance
from trustworthai.analysis.evaluation_metrics.challenge_metrics import per_model_chal_stats, write_model_metric_results, avd_mean_analysis, dice_mean_analysis, dice_across_samples_analysis

from trustworthai.utils.uncertainty_maps.entropy_map import entropy_map_from_samples
from trustworthai.utils.data_preprep.dataset_to_list import dataset_to_list
from trustworthai.utils.plotting.save_samples_and_umaps import plot_and_save_samples_and_umaps
from trustworthai.utils.plotting.sample_diversity_plots import sample_diversity_plot


from trustworthai.analysis.calibration.calibration_over_samples import calibration_over_samples
from trustworthai.analysis.pavpu.record_pavpu_results import pavpu_analysis
from trustworthai.analysis.connected_components.connected_comps_2d import conn_comp_2d_analysis
from trustworthai.analysis.evaluation_metrics.sUEO import sUEO_per_individual_analysis, UEO_per_threshold_analysis
from trustworthai.analysis.evaluation_metrics.VVC import VVC_corr_coeff
from trustworthai.analysis.sample_diversity.sample_distances import sample_diverity_analysis
from trustworthai.analysis.prediction_type_analysis.tp_fp_fn_analysis import tp_fp_fn_analysis

from trustworthai.utils.print_and_write_func import print_and_write

print("banana")

def evaluation_pipeline(model, dataset, temperature, results_dir, model_name, dataset_stride=1, num_samples=20):
    do_normalize=True
    
    # load data
    xs3d, ys3d = dataset_to_list(dataset, dataset_stride)
    # ima little concerned because I have swapped the axes for the mean and samples but not for xs3d and ys3d so its all a bit doj aha

    # load model outputs and u-map
    means3d, samples3d = model.sample_over_3Ddataset(dataset, dataset_stride=dataset_stride, rsample=True, temperature=temperature, num_samples=num_samples)
    ind_ent_maps = [entropy_map_from_samples(samples3d[i], do_normalize=do_normalize) for i in range(len(ys3d))]

    # setup folder and files
    try:
        save_folder = results_dir + model_name + "/results/"
        imgs_folder = save_folder + "images/"
        os.mkdir(save_folder)
        os.mkdir(imgs_folder)
    except:
        raise ValueError(f"Results folder for model name {model_name} already EXISTS, do not want to overwrite!!")
        
    text_results_file = save_folder + "text_results.txt"
    print_and_write(text_results_file, "results for " + model_name, clear_file=True)

    # 1 plot some selected samples and umaps
    # scan_ids = [-1,-1,3,3,27,27,0,0]
    # slice_ids = [38,26,32,18,33,37,22,24]
    scan_ids = [0,1,2,3,4]
    slice_ids = [31, 30, 30, 55, 55]
    
    plot_and_save_samples_and_umaps(save_folder, samples3d, ys3d, ind_ent_maps, scan_ids, slice_ids)

    # 2 compute the challenge dataset metrics and save them to file
    data = per_model_chal_stats(means3d, ys3d)
    write_model_metric_results(text_results_file, data)
    

    # 3 avd, dice, and dice per sample for each individual # TODO CHECK ITS CORRECT
    # compute the dice per sample, per individual
    # TODO: check whether this code is actually correct, like is it not
    # [s][ind] not [ind][s]
    avd_mean_analysis(text_results_file, means3d, ys3d)
    dice_mean_analysis(text_results_file, means3d, ys3d)
    tensor_alldice3d = dice_across_samples_analysis(text_results_file, ys3d, samples3d)
    
    plt.style.use('fivethirtyeight')

    # sample div plot
    sample_diversity_plot(save_folder, tensor_alldice3d, "Dice")
    
    # calibration
    calibration_over_samples(save_folder, text_results_file, means3d, samples3d, ys3d, do_normalize)

    # SAMPLE DIVERSITY AND GENERALIZED ENERGY DISTANCE
    sample_diverity_analysis(text_results_file, samples3d, ys3d)

    # COMPUTE PAVPU METRICS
    pavpu_analysis(save_folder, text_results_file, means3d, samples3d, ys3d, ind_ent_maps, uncertainty_thresholds=torch.arange(0, 0.7, 0.01), accuracy_threshold=0.9, window_size=16, do_normalize=do_normalize)

    # QUALITY CONTROL IN 3D - vcc corr coeff
    # TODO@ I'm really not sure this method even makes any sense....
    VVC_corr_coeff(text_results_file, ys3d, samples3d, do_normalize)

    # TP TN FN DISTRIBUTION!!
    tp_fp_fn_analysis(save_folder, text_results_file, xs3d, means3d, ys3d, samples3d, ind_ent_maps)

    ### MISSING LESIONS IN 2D SLICES
    conn_comp_2d_analysis(save_folder, text_results_file, uncertainty_thresholds, ys3d, means3d, ind_ent_maps)

    ### EXAMINATION OF SHAPE BASED METRICS
    sUEO_per_individual_analysis(text_results_file, ys3d, ind_ent_maps)

    # UEO = sUEO but U is thresholded binary now. We can plot it over tau, increasing in 0.05 steps
    UEO_per_threshold_analysis(save_folder, text_results_file, uncertainty_thresholds, ys3d, ind_ent_maps)
    
    print("DONE")
