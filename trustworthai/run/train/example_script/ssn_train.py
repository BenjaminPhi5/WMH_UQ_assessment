print("strawberry")

# loss function and metrics
from trustworthai.utils.losses_and_metrics.dice_loss import DiceLossWithWeightedEmptySlices
from trustworthai.utils.losses_and_metrics.dice_loss_metric import DiceLossMetric, SsnDiceMeanMetricWrapper

# predefined training dataset
from trustworthai.utils.data_preprep.dataset_pipelines import load_data

# fitter
from trustworthai.utils.fitting_and_inference.fitters.basic_lightning_fitter import StandardLitModelWrapper
from trustworthai.utils.fitting_and_inference.get_trainer import get_trainer

# model
from trustworthai.run.model_load.load_ssn import load_ssn

# optimizer and lr scheduler
import torch

# misc
import argparse
import os
import shutil

print("banana")

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
    
    # model specific parameters
    parser.add_argument('--ssn_rank', default=15, type=int)
    parser.add_argument('--ssn_epsilon', default=1e-5, type=float)
    parser.add_argument('--ssn_mc_samples', default=10, type=int)
    parser.add_argument('--ssn_sample_dice_coeff', default=0.05, type=float)
    parser.add_argument('--ssn_pre_head_layers', default=16, type=int)
    
    # general arguments for the loss function
    parser.add_argument('--dice_factor', default=5, type=float)
    parser.add_argument('--xent_factor', default=0.01, type=float)
    parser.add_argument('--dice_empty_slice_weight', default=0.5, type=float)
    
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
    parser.add_argument('--no_test_fold', default='false', type=str)
    
    return parser


def main(args):
    
    print(args.cross_validate, type(args.cross_validate))
    
    # TODO: I need to build the model name somehow...
    model_dir = os.path.join(args.ckpt_dir, args.model_name) # TODO model name dir goes here
    
    if os.path.exists(model_dir):
        if not args.overwrite:
            raise ValueError(f"model directly ALREADY EXISTS: do not wish to overwrite!!: {model_dir}")
        else:
            print("warning, folder being overwritten")
            shutil.rmtree(model_dir)
            os.mkdir(model_dir)
    
    # get the 2d axial slice dataloaders
    train_dl, val_dl, test_dl = load_data(
        dataset=args.dataset, 
        test_proportion=args.test_split, 
        validation_proportion=args.val_split,
        seed=args.seed,
        empty_proportion_retained=args.empty_slice_retention,
        batch_size=args.batch_size,
        dataloader2d_only=True,
        cross_validate=args.cross_validate,
        cv_split=args.cv_split,
        cv_test_fold_smooth=args.cv_test_fold_smooth,
        merge_val_test=args.no_test_fold # causes the val and test fold to be merged
    )
    
    model_raw, loss = load_ssn(args)

    # setup optimizer and model wrapper
    optimizer_params={"lr":args.lr, "weight_decay":args.weight_decay}
    optimizer = torch.optim.Adam
    lr_scheduler_params={"milestones":[1000], "gamma":0.5}
    lr_scheduler_constructor = torch.optim.lr_scheduler.MultiStepLR

    # wrap the model in the pytorch_lightning module that automates training
    model = StandardLitModelWrapper(model_raw, loss, 
                                    logging_metric=lambda : None,
                                    optimizer_params=optimizer_params,
                                    lr_scheduler_params=lr_scheduler_params,
                                    optimizer_constructor=optimizer,
                                    lr_scheduler_constructor=lr_scheduler_constructor
                                   )
   
    # train the model
    trainer = get_trainer(args.max_epochs, model_dir, early_stop_patience=args.early_stop_patience)
    trainer.fit(model, train_dl, val_dl)
    
    # get best checkpoint based on loss on validation data
    try:
        #"save best model checkpoint name"
        with open(os.path.join(model_dir, "best_ckpt.txt"), "w") as f:
            f.write(trainer.checkpoint_callback.best_model_path)
            f.write("\n")
            for key , value in vars(args).items():
                f.write(f"{key}: {value}\n")
            
        trainer.validate(model, val_dl, ckpt_path='best')
    except:
        print("failed to run validate to print best checkpoint path oh well")


if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
