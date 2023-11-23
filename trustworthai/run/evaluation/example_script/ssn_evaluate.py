#TODO! put in all the evaluation scripts as their own functions in the evaluate folder
# make each evaluation its own module.
print("strawberry")

# predefined training dataset
from trustworthai.utils.data_preprep.dataset_pipelines import load_data

# model
from trustworthai.run.model_load.load_ssn import load_ssn
from trustworthai.run.evaluation.load_best_checkpoint import load_best_checkpoint

# eval pipeline
from trustworthai.run.evaluation.evaluation_pipeline import evaluation_pipeline

# misc
import argparse

print("banana")

def construct_parser():
    parser = argparse.ArgumentParser(description = "train models")
    
    # folder arguments
    parser.add_argument('--ckpt_dir', default=None, type=str)
    parser.add_argument('--dice_factor', default=None, type=int)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--results_dir', default=None, type=str)
    
    # data generation arguments
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--test_split', default=0.15, type=float)
    parser.add_argument('--val_split', default=0.15, type=float)
    parser.add_argument('empty_slice_retention', default=0.1, type=float)
    
    # model specific parameters
    parser.add_argument('--ssn_rank', default=15, type=int)
    parser.add_argument('--ssn_epsilon', default=1e-5, type=int)
    parser.add_argument('--ssn_mc_samples', default=10, type=int)
    parser.add_argument('--ssn_sample_dice_coeff', default=0.05, type=float)
    parser.add_argument('--ssn_pre_head_layers', default=16, type=int)
    
    # general arguments for the loss function
    parser.add_argument('--dice_factor', default=5. type=float)
    parser.add_argument('--xent_factor', default=0.01, type=float)
    parser.add_argument('--dice_empty_slice_weight', default=0.5, type=float)
    
    # training paradigm arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--results_dir', default='s2208943/results/revamped_models/', type=str)
    parser.add_argument('--dropout_p', default=0.0, type=float)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--early_stop_patience', default=15, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    
    return parser


def main(args):
    
    # todo: solve how to determine the model name (e.g from the args??)
    model_folder = os.path.join(args.ckpt_dir, args.model_name)
    
    # get the 3d un-augmented datasets
    _, _, test_ds = load_data(
        dataset="ed", 
        test_proportion=args.test_split, 
        validation_proportion=args.val_split,
        seed=args.seed,
        empty_proportion_retained=args.empty_slice_retention,
        batch_size=args.batch_size,
        dataset3d_only=True
    )
    
    # get the 3d un-augmented datasets
    _, _, test_ds_new_domain = load_data(
        dataset="chal", 
        test_proportion=args.test_split, 
        validation_proportion=args.val_split,
        seed=args.seed,
        empty_proportion_retained=args.empty_slice_retention,
        batch_size=args.batch_size,
        dataset3d_only=True
    )
    
    # load the model
    model_raw, loss = load_ssn(args)
    
    # load the best checkpoint
    model = load_best_checkpoint(model, loss, model_folder)
    
    # do evaluation
    temperature = 1.5
    # TODO: uncomment once this is running on the cluster!
    #evaluation_pipeline(model, test_ds, temperature, args.results_dir, args.model_name, dataset_stride=1)
    evaluation_pipeline(model, test_ds_new_domain, temperature, args.results_dir, args.model_name + "_DOMAIN", dataset_stride=1, num_samples=20)