"""
so to make this work I need to

4) run it for each fold
5) run each model 3 times over. Nice.
6) change model checkpoint dir
7) setup results file with target and split name and whether it uses umap or not

"""

# trainer
print("strawberry")
from trustworthai.utils.fitting_and_inference.fitters.basic_lightning_fitter import StandardLitModelWrapper
from trustworthai.utils.fitting_and_inference.get_trainer import get_trainer

# data
from trustworthai.utils.data_preprep.dataset_pipelines import load_clinscores_data, load_data, ClinScoreDataRetriever
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from trustworthai.utils.augmentation.augmentation_pipelines import get_transforms
from trustworthai.utils.data_preprep.dataset_wrappers import *

# model
from trustworthai.models.modified_resnet import *

# packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchinfo import summary
from tqdm import tqdm
from collections import defaultdict
from natsort import natsorted
import torchmetrics
from trustworthai.utils.print_and_write_func import print_and_write
import argparse


print("banana")


###################################
############## Loading extra inputs (umap and model seg)
umap_model_name = "SSN_Ens"
output_maps_dir = f"/home/s2208943/ipdis/data/preprocessed_data/EdData_output_maps/{umap_model_name}/"

output_maps_files = os.listdir(output_maps_dir)
def get_output_maps_for_ds(ds):
    output_maps_lists = defaultdict(lambda : [])
    for data in tqdm(ds, position=0, leave=True):
        ID = data[2]['ID']
        output_maps_data = np.load(f"{output_maps_dir}{ID}_out_maps.npz")
        for output_type in output_maps_data.keys():
            output_maps_lists[output_type].append(torch.from_numpy(output_maps_data[output_type]))
            
    return output_maps_lists


### building the combined datasets with 
# augmentation transforms, added in umaps and predictions as channels
# and code to select 3 slices from the image.

def build_clinscores_prediction_dataset(ds, t=2.5, v=3, clin_fields=['age'], target_field='DWMH'):
    
    # sort out the extra fields I need (e.g smoking one hot encoded)
    ds = CustomClinFieldsDataset(ds)
    
    # load in the extra data (e.g umaps) to build a combined dataset
    #print("loading umaps and wmh preds images")
    output_maps_test = get_output_maps_for_ds(ds)
    added_channels_ds = AddChannelsDataset(ds, output_maps_test)
    
    # compute which slices have wmh burdern with 3 std of the max
    slices_within_std_for_ds = get_slices_within_std_for_ds(ds, t=t)
    
    # get dataset that randomly samples three slices for each input from within the range of slices within one std
    new_ds = SlicesDataset(added_channels_ds, slices_within_std_for_ds, v=v, transform=get_transforms())
    
    # combine the clin scores data into the x
    new_ds = ClinicalDataset(new_ds, clin_fields, target_field)
    
    new_ds = NonNanDataset(new_ds)
    
    return new_ds


### selecting a model
class PredictionModel(torch.nn.Module):
    # uses a modified resnet that bypasses the usual 
    def __init__(self, include_umaps, num_slices, num_clin_features, out_classes, latent_fc_features=64):
        super().__init__()
        model_base = resnet18()
        
        if include_umaps:
            image_channels = 3 # flair, seg, umap
        else:
            image_channels = 2 # flair, seg
        self.include_umaps = include_umaps
        
        model_base.conv1 = nn.Conv2d(image_channels*num_slices, model_base.init_inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.model_base = model_base
        
        # replace the head of the model with another layer.
        
        self.fc1 = nn.Linear(model_base.fc.in_features + num_clin_features, latent_fc_features)
        self.a = nn.ReLU()
        self.fc2 = nn.Linear(latent_fc_features, out_classes)
        
    def forward(self, inp):
        
        xid = {"flair":[0,1,2], "mask": [3,4,5], "t1":[6,7,8], "var": [9,10,11], "ent":[12,13,14], "pred":[15,16,17], "seg":[18,19,20]}
        
        x = inp[0]
        
        # x = x[:,[*xid["flair"], *xid["pred"], *xid["ent"]]] 
        # x = x[:,[*xid["flair"], *xid["pred"], *xid["ent"], *xid["var"]]]
        
        if self.include_umaps:
            x = x[:,[*xid["flair"], *xid["pred"], *xid["ent"]]] 
        else:
            x = x[:,[*xid["flair"], *xid["pred"]]]
        # x = None
        
        #print(x.shape, [*xid["flair"], *xid["pred"]])
        
        clin_data = inp[1]
        
        if x != None:
            features = self.model_base(x)
        else:
            features = torch.zeros(inp[0].shape[0], 512).cuda()
        dense_input = torch.cat([features, clin_data], dim=1)
        
        #print
        
        out = self.fc2(self.a(self.fc1(dense_input)))
        
        return out
        

class xent_wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_loss = torch.nn.CrossEntropyLoss()
    def forward(self, y_hat, y):
        return self.base_loss(y_hat, y.type(torch.long))
    
    
def construct_parser():
    parser = argparse.ArgumentParser(description = "train models")
    
    # folder arguments
    parser.add_argument('--target', default="DWMH", type=str)
    parser.add_argument('--split', default=0, type=int)
    parser.add_argument('--include_umaps', default="False", type=str)
    
    return parser


def main(args):
    
    target = args.target
    split = args.split
    include_umaps = args.include_umaps
    
    include_umaps = True if include_umaps == "True" else False
    
    save_folder = f"/home/s2208943/ipdis/UQ_WMH_methods/trustworthai/run/fazekas_pred_results/"
    results_file = save_folder + f"{target}_{split}_umaps_{include_umaps}_results.txt"
    
    
    if target == "DWMH" or target == "PVWMH":
        num_classes = 4
    elif target == "total_fazekas" or target == "supAtrophy" or target == "deepAtrophy":
        num_classes = 7 # includes the 6 class.
    else:
        num_classes = 2 # for stroke, or tab fazekas, or the other tab variables etc.
    
    ### load the data
    clin_retriever = ClinScoreDataRetriever(use_updated_scores=True)
    
    train_ds_clin, val_ds_clin, test_ds_clin = clin_retriever.load_clinscores_data(
            combine_all=False,
            test_proportion=0.15, 
            validation_proportion=0.12,
            seed=3407,
            cross_validate=True,
            cv_split=split,
            cv_test_fold_smooth=1,

        )
    
    t = 2.5
    v = 3
    # clin_fields = ['age', 'sex', 'diabetes']
    # clin_fields = ['age', 'sex', 'diabetes', 'hypertension', 'hyperlipidaemia', 'smoking_0', 'smoking_1', 'smoking_2', 'ICV']
    clin_fields = ['age', 'sex', 'diabetes', 'hypertension', 'hyperlipidaemia', 'smoking_0', 'smoking_1', 'smoking_2']
    new_test_ds = build_clinscores_prediction_dataset(test_ds_clin, t=t, v=v, clin_fields=clin_fields, target_field=target)
    new_val_ds = build_clinscores_prediction_dataset(val_ds_clin, t=t, v=v, clin_fields=clin_fields, target_field=target)
    new_val_ds = RepeatDataset(new_val_ds, repeats=15) # to give a less noisy update on the validation dataloader
    new_train_ds = build_clinscores_prediction_dataset(train_ds_clin, t=t, v=v, clin_fields=clin_fields, target_field=target)
    
    ### defining the dataloader
    batch_size = 12
    train_dataloader = DataLoader(new_train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(new_val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(new_test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # load model and train the model 3 times over
    loss = xent_wrapper()
    print(include_umaps, type(include_umaps))
    for m in range(5):
        print(f"training model: {m}")
        
        model_raw = PredictionModel(include_umaps=include_umaps, num_slices=3, num_clin_features=8, out_classes=num_classes)#.cuda()
        
        weight_decay = 0.0001
        max_epochs = 100
        lr=2e-4
        early_stop_patience = 15

        optimizer_params={"lr":lr, "weight_decay":weight_decay}
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
        trainer = get_trainer(max_epochs, "/disk/scratch_big/s2208943/results/", early_stop_patience=early_stop_patience)

        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.validate(model, val_dataloader, ckpt_path='best')
        
        print_and_write(results_file, f"first layer check: {model.model.model_base.conv1}")

        # perform evaluation
        print("performing evaluation!!!")
        f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)
    
        # since the test dataloader does not have shuffle, its actually easy to take the
        # mode...
        # IMPORTANT ASSUMES TEST DATALOADER HAS SHUFFLE TURNED OFF!!
        all_y_hats = []
        accs = []
        top2_accs = []

        batch_accs = []

        runs = 50
        model.eval()

        for _ in tqdm(range(runs), position=0, leave=True):
            y_hat_preds = []

            for i, ((x, clin_data), y) in enumerate(test_dataloader):
                with torch.no_grad():
                    pred = torch.nn.functional.softmax(model.cuda()((x.cuda(), clin_data.cuda())), dim=1).cpu()
                    y_hat = pred.argmax(dim=1)

                    y_hat_preds += list(y_hat)

                    acc = (y == y_hat).sum() / len(y)
                    accs.append(acc)

                    top2 = pred.topk(dim=1, k=2)[1]
                    top2_acc = ((y == top2[:,0]) | (y == top2[:,1])).sum() / len(y)
                    top2_accs.append(top2_acc)

            all_y_hats.append(y_hat_preds)

        # calculate the mode accuracy (which is usually higher than mean over multiple runs but can vary quite a bit...
        print(torch.Tensor(all_y_hats).shape)
        mode_preds = torch.Tensor(all_y_hats).mode(dim=0)[0]
        all_ys = []
        for i, ((x, clin_data), y) in enumerate(test_dataloader):
            all_ys += list(y)
        all_ys = torch.Tensor(all_ys)

        mode_acc = (mode_preds == all_ys).sum() / len(all_ys)
        runs_mean_acc = torch.Tensor(accs).mean()
        runs_mean_top2 = torch.Tensor(top2_accs).mean()

        # f1 on mode predictions
        mode_f1 = f1_metric(mode_preds, all_ys)
        
        
        if m == 0:
            print_and_write(results_file, "target ys", newline=1)
            print_and_write(results_file, all_ys)
        
        # headline stats
        print_and_write(results_file, f"for model {m}: mode acc: {mode_acc}, run mean acc: {runs_mean_acc}, top-2 acc:{runs_mean_top2}, mode f1: {mode_f1}")
        
        # per individual mode predictions
        print_and_write(results_file, f"model {m} mode preds", newline=1)
        print_and_write(results_file, mode_preds)
        # per individual every prediction
        print_and_write(results_file, f"model {m} all preds", newline=1)
        print_and_write(results_file, torch.Tensor(all_y_hats).tolist())
        
        
if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)