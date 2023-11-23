# trainer
print("strawberry")
from trustworthai.utils.fitting_and_inference.fitters.basic_lightning_fitter import StandardLitModelWrapper
from trustworthai.utils.fitting_and_inference.get_trainer import get_trainer

# data
from twaidata.torchdatasets.MRI_3D_nolabels_inram_ds import MRI_3D_nolabels_inram_ds

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
import argparse
print("banana")

def construct_parser():
    parser = argparse.ArgumentParser(description = "train models")
    parser.add_argument('--label', default='WMH_Deep', type=str)
    parser.add_argument('--channel_id', default='ent', type=str)
    parser.add_argument('--seed_index', default=0, type=int)
    parser.add_argument('--latent_layers', default=3, type=int)
    parser.add_argument('--layer_div', default= 8, type=int)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--reweighted', default="true", type=str)
    parser.add_argument('--dummy_run', default="false", type=str)
    
    return parser

def get_output_maps_for_ds(output_maps_dir, ids):
    output_maps_lists = defaultdict(lambda : [])
    key_order = None
    for ID in tqdm(ids, position=0, leave=True):
        output_maps_data = np.load(f"{output_maps_dir}{ID}_out_maps.npz")
        key_order = natsorted(list(output_maps_data.keys()))
        for output_type in key_order:
            output_maps_lists[ID].append(torch.from_numpy(output_maps_data[output_type]))

    return output_maps_lists, key_order

def normalize_column(df, column_id):
    # normalizes a column (not inlcuding nan values)
    values = df[column_id].values.astype(np.float32)
    values = values[~np.isnan(values)]
    mean = values.mean()
    std = values.std()
    df[column_id] = (df[column_id].values - mean)/std

def one_hot_encode(df, field, remove_original=True):
    """computes a one hot encoding of the given field"""

    onehot_cols = pd.get_dummies(df[field], prefix=field)

    if remove_original:
        df = df.drop(columns=[field], inplace=False)

    df = pd.concat([df, onehot_cols], axis=1)

    return df

def convert_symbol_to_nan(df, field, symbol):
    df = df.copy()
    """
    converts all inputs conforming to 'symbol' to np.nan for the given 'field' in the 'df'
    e.g if symbol=' ' and field='totalChl' then any instances of ' ' in the 'totalChl' column will be replaced with np.nan
    """
    values = df[field].values
    locs = values == symbol
    values[locs] = np.nan

    df[field] = values

    return df

def filter_rows_with_nans(df, field, inplace=False):
    """
    removes all rows from df which have nan for the given field value
    """
    values = df[field].values.astype(np.float32)
    nan_locs = np.where(np.isnan(values))[0]
    df = df.drop(nan_locs, inplace=inplace)
    df = df.reset_index(drop=True)

    return df

def prepare_ADNI_dfs(
    ratings_df, variables_df,
    selected_columns=[
        'Patient ID', 'AGE', 'Ventricles_bl %', 'Hippocampus_bl %',
        'WholeBrain_bl %', 'Entorhinal_bl %', 'Fusiform_bl %',
        'MidTemp_bl %', 'BMI', 'DX.bl', 'CV RISK FACTORS', 'APOE4',
        'WMH_PV', 'WMH_Deep', 'Total', 'PTGENDER', 'E-M RISK FACTORS',
    ]):

    r_df = ratings_df.copy()
    v_df = variables_df.copy()

    # in the variables df, put all the column headings actually in the heading,
    variables_heading_map = {
        key:column_heading 
        for (column_heading, key) in v_df.iloc[0].items()
    }
    for key, column_heading in variables_heading_map.items():
        v_df[key] = v_df[column_heading].values
        v_df = v_df.drop(columns=[column_heading], inplace=False)


    # remove any * characters and ' ' from patient IDs
    r_df['Patient ID'] = [str(pid).replace('*', '').replace(' ', '') for pid in r_df['Patient ID'].values]
    v_df['Patient ID'] = [str(pid).replace('*', '').replace(' ', '') for pid in v_df['Patient ID'].values]

    # remove any rows that do not have a patient ID. patient ID can be detected due to having a '_S_' string in it.
    pid_locs_rdf = ['_S_' in pid for pid in r_df['Patient ID'].values]
    pid_locs_vdf = ['_S_' in pid for pid in v_df['Patient ID'].values]
    r_df = r_df.loc[pid_locs_rdf]
    v_df = v_df.loc[pid_locs_vdf]

    # join the two dataframes
    df = pd.merge(r_df, v_df, how='left')

    print(df.keys())

    # drop any column that isn't selected
    df = df[selected_columns]

    # normalize columns
    for norm_column in ['AGE', 'Ventricles_bl %', 'Hippocampus_bl %', 'WholeBrain_bl %', 'Entorhinal_bl %', 'Fusiform_bl %', 'MidTemp_bl %', 'BMI', 'PTEDUCAT', 'PTRACCAT']:
        if norm_column in selected_columns:
            normalize_column(df, norm_column)

    # one hot encoder columns
    for one_hot_col in ['DX.bl', 'CV RISK FACTORS', 'APOE4']:
        if one_hot_col in selected_columns:
            df = one_hot_encode(df, one_hot_col)

    # set values of zero to nan for brain measurement fields
    for no_zero_col in ['Ventricles_bl %', 'Hippocampus_bl %', 'WholeBrain_bl %', 'Entorhinal_bl %', 'Fusiform_bl %', 'MidTemp_bl %']:
        df = convert_symbol_to_nan(df, no_zero_col, 0)

    # change PTGENDER column to 0,1 (as opposed to 1, 2)
    df['PTGENDER'] = df['PTGENDER'] - 1

    # in all selected columns, drop rows that contain a nan value
    for col in df.keys():
        if col not in ['Patient ID']:
            try:
                df = filter_rows_with_nans(df, col)
            except:
                print("failed on: ", col)

    return df
    
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from trustworthai.utils.data_preprep.splits import cross_validate_split

### defining the augmentation procedure
from trustworthai.utils.augmentation.standard_transforms import (
    RandomFlip, GaussianBlur, GaussianNoise,
    RandomResizeCrop, RandomAffine,
    NormalizeImg, PairedCompose, LabelSelect,
    PairedCentreCrop, CropZDim,
)
import torch


def get_transforms():
    transforms = [
        LabelSelect(label_id=1),
        RandomFlip(p=0.5, orientation="horizontal"),
        # GaussianBlur(p=0.5, kernel_size=7, sigma=(.1, 1.5)),
        # GaussianNoise(p=0.2, mean=0, sigma=0.2),
        RandomAffine(p=0.2, shear=(-18,18)),
        RandomAffine(p=0.2, degrees=15),
        RandomAffine(p=0.2, translate=(-0.1,0.1)),
        RandomAffine(p=0.2, scale=(0.9, 1.1)),
#         #RandomResizeCrop(p=1., scale=(0.6, 1.), ratio=(3./4., 4./3.))

#         #RandomResizeCrop(p=1., scale=(0.3, 0.5), ratio=(3./4., 4./3.)) # ssn
    ]
    transforms.append(lambda x, y: (x, y.squeeze().type(torch.long)))
    return PairedCompose(transforms)

    ### combine the clinical scores data into the x information.
    # generated with chatgpt
class ClinicalDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, fields, target_field):
        self.base_dataset = base_dataset
        self.fields = fields
        self.target_field = target_field

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, y, clin_data = self.base_dataset[index]
        clin_data_fields = clin_data[self.fields].values
        clin_data_tensor = torch.from_numpy(clin_data_fields.astype(np.float32))
        target_field = clin_data[self.target_field]
        return (x, clin_data_tensor), target_field

    # torch dataset that filters out nans
class NonNanDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.indices = []

        for i in range(len(self.original_dataset)):
            (x, clin_data), y = self.original_dataset[i]
            if not (np.isnan(y) or torch.any(torch.isnan(clin_data))):
                self.indices.append(i)

    def __getitem__(self, index):
        original_index = self.indices[index]
        return self.original_dataset[original_index]

    def __len__(self):
        return len(self.indices)

class RepeatDataset(Dataset):
    def __init__(self, original_dataset, repeats):
        self.original_dataset = original_dataset
        self.repeats=repeats

    def __getitem__(self, idx):
        return self.original_dataset[idx % len(self.original_dataset)]

    def __len__(self):
        return len(self.original_dataset) * self.repeats

class AddChannelsDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, extra_x_channels_lists, IDs):
        self.base_dataset = base_dataset
        self.extra_x_channels_lists = extra_x_channels_lists
        self.IDs = IDs

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        ID = self.IDs[idx]
        x = data[0]
        # print(x.shape)
        # print(torch.stack(self.extra_x_channels_lists[ID]).shape)
        x = torch.cat([x, torch.stack(self.extra_x_channels_lists[ID])], dim=0)

        return (x, *data[1:], ID)

    def __len__(self):
        return len(self.base_dataset)

class ExtractYChannelDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, y_channel_idx):
        self.base_dataset = base_dataset
        self.y_channel_idx = y_channel_idx

        self.x_channels = [i for i in range(base_dataset[0].shape[0])]
        self.x_channels.remove(y_channel_idx)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        return data[self.x_channels], data[self.y_channel_idx].unsqueeze(0)

    def __len__(self):
        return len(self.base_dataset)

class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, transforms):
        self.base_dataset = base_dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        (x, y, clin_data), label = self.base_dataset[idx]
        return (*self.transforms(x, y), clin_data), label

    def __len__(self):
        return len(self.base_dataset)

class ImgsAndDfDataset(Dataset):
    def __init__(self, base_dataset, df, selected_fields, label_field):
        self.base_dataset = base_dataset
        self.df = df
        self.selected_fields = selected_fields
        self.label_field = label_field

    def __getitem__(self, idx):
        x, y, ID = self.base_dataset[idx]
        clin_data = self.df.loc[self.df['Patient ID'] == "_".join(ID.split('_')[1:-1])]
        fields = clin_data[self.selected_fields].values

        # print(len(fields))
        label = clin_data[self.label_field].values[0]

        return (x, y, fields), label

    def __len__(self):
        return len(self.base_dataset)

class SkipBadIndexesDs(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        valid_indexes = []

        for idx in tqdm(range(len(base_dataset)), position=0, leave=True):
            try:
                _ = base_dataset[idx]
                valid_indexes.append(idx)
            except IndexError:
                continue

        self.valid_indexes = valid_indexes

    def __getitem__(self, idx):
        return self.base_dataset[self.valid_indexes[idx]]

    def __len__(self):
        return len(self.valid_indexes)

class ConsumedDataset(Dataset):
    def __init__(self, base_dataset):
        self.elements = [data for data in tqdm(base_dataset, position=0, leave=True)]

    def __getitem__(self, idx):
        return self.elements[idx]

    def __len__(self):
        return len(self.elements)

        
import torchvision as tv

class FormatDataset(Dataset):
    def __init__(self, base_dataset, slice_range=(25,45), centre_crop=(220,160), channels=['flair']):
        self.base_dataset = base_dataset
        self.slice_range = slice_range
        self.centre_crop = centre_crop
        self.channels = channels

    def __getitem__(self, idx):
        channels_id = {
            "flair":0,
            "t1":1,
            "ent":2,
            "pred":3,
            "seg":4,
            "var":5,
        }

        (x_3d, mask, clin_data), label = self.base_dataset[idx]

        # print(x_3d.shape)

        selected_channels = [channels_id[c] for c in self.channels]

        # Select the slices from the 3D image
        x_3d = x_3d[selected_channels]
        # print(x_3d.shape)

        slices = np.arange(self.slice_range[0], self.slice_range[1], 1)
        x_3d = x_3d[:, slices]
        # print(x_3d.shape)

        # Reshape the slices into C*v 2D tensors
        x_2d = torch.reshape(x_3d, (-1, x_3d.shape[-2], x_3d.shape[-1]))
        # x_2d = x_3d

        # print(x_2d.shape)

        x_2d = tv.transforms.functional.center_crop(x_2d, self.centre_crop)

        # print(x_2d.shape)

        return (x_2d, clin_data.astype(np.float32).squeeze()), label


    def __len__(self):
        return len(self.base_dataset)

### SETUP MODEL FOR TRAINING
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        channel_size_divide=1,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64//channel_size_divide
        self.init_inplanes = self.inplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64//channel_size_divide, layers[0])
        self.layer2 = self._make_layer(block, 128//channel_size_divide, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256//channel_size_divide, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512//channel_size_divide, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights,#: Optional[WeightsEnum],
    progress: bool,
    channel_size_divide,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, channel_size_divide, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def resnet18(*, progress: bool = True, channel_size_divide=2, **kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [2, 2, 2, 2], None, progress, channel_size_divide=channel_size_divide, **kwargs)

def resnet34(*, progress: bool = True, **kwargs: Any) -> ResNet:

    return _resnet(BasicBlock, [3, 4, 6, 3], None, progress, **kwargs)

### selecting a model

class PredictionModel(torch.nn.Module):
    # uses a modified resnet that bypasses the usual 
    def __init__(self, image_channels, num_clin_features, out_classes, latent_fc_features=64, channel_size_divide=2, finetune_head=False):
        super().__init__()
        model_base = resnet18(channel_size_divide=channel_size_divide)
        self.channel_size_divide = channel_size_divide
        model_base.conv1 = nn.Conv2d(image_channels, model_base.init_inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.model_base = model_base
        self.finetune_head = finetune_head

        # replace the head of the model with another layer.
        self.fc1 = nn.Linear(model_base.fc.in_features//channel_size_divide + num_clin_features, latent_fc_features)
        self.a = nn.ReLU()
        self.fc2 = nn.Linear(latent_fc_features, out_classes)

    def forward(self, inp):


        x = inp[0]
        clin_data = inp[1]

        # x = None

        if x != None:
            if not self.finetune_head:
                features = self.model_base(x)
            else:
                with torch.no_grad():
                    features = self.model_base(x)
        else:
            features = torch.zeros(inp[0].shape[0], 512//self.channel_size_divide).cuda()
        dense_input = torch.cat([features, clin_data], dim=1)

        out = self.fc2(self.a(self.fc1(dense_input)))

        return out

class PredictionModelNHeadLayer(torch.nn.Module):
    # uses a modified resnet that bypasses the usual 
    def __init__(self, image_channels, num_clin_features, out_classes, latent_fc_features=64, channel_size_divide=2, finetune_head=False, n_heads=3):
        super().__init__()
        model_base = resnet18(channel_size_divide=channel_size_divide)
        self.channel_size_divide = channel_size_divide
        model_base.conv1 = nn.Conv2d(image_channels, model_base.init_inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.model_base = model_base
        self.finetune_head = finetune_head

        self.a = nn.ReLU()
        if n_heads > 1:
            # replace the head of the model with another layer.
            self.fc1 = nn.Linear(model_base.fc.in_features//channel_size_divide + num_clin_features, latent_fc_features)
            self.extra_heads = nn.ModuleList([nn.Linear(latent_fc_features, latent_fc_features) for i in range(n_heads - 2)] + [nn.Linear(latent_fc_features, out_classes)])

        else:
            # replace the head of the model with another layer.
            self.fc1 = nn.Linear(model_base.fc.in_features//channel_size_divide + num_clin_features, out_classes)
            self.extra_heads = None

        if self.extra_heads:
            print(self.extra_heads)

    def forward(self, inp):


        x = inp[0]
        clin_data = inp[1]

        # x = None

        if x != None:
            if not self.finetune_head:
                features = self.model_base(x)
            else:
                with torch.no_grad():
                    features = self.model_base(x)
        else:
            features = torch.zeros(inp[0].shape[0], 512//self.channel_size_divide).cuda()
        dense_input = torch.cat([features, clin_data], dim=1)

        out = self.fc1(dense_input)

        if self.extra_heads:
            for h in self.extra_heads:
                out = h(self.a(out))

        return out

class xent_wrapper(nn.Module):
    def __init__(self, reweighted, weight=None):
        super().__init__()
        if reweighted:
            self.base_loss = torch.nn.CrossEntropyLoss(weight = weight)
        else:
            self.base_loss = torch.nn.CrossEntropyLoss()
    def forward(self, y_hat, y):
        return self.base_loss(y_hat, y.type(torch.long))

import torchmetrics
f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=4) # note need to set the F1 score correctly depending on the number of classes in the task!!

def main(args):
    genner = torch.Generator().manual_seed(torch.randint(size=(1,), low=0, high=1000).item())
    
    # make sure that I get the label correct for each dataset respectively.
    label = args.label
    
    channel_id = args.channel_id
    if channel_id == "both":
        channels = ['ent', 'pred']
    else:
        channels = [channel_id]
    sid = args.seed_index
    seed = [128, 255, 693, 19, 385][sid]
    latent_layers = args.latent_layers
    layer_div = args.layer_div
    weight_decay = args.weight_decay
    reweighted = True if args.reweighted.lower() == "true" else False 
    reweighted_id = args.reweighted.lower()
    dummy_run = True if args.dummy_run.lower() == "true" else False
    
    if os.path.exists(f"/home/s2208943/ipdis/UQ_WMH_methods/trustworthai/run/ADNI_300_analysis/new_fazekas_prediction_code/results/ADNI/ADNI_{label}_{channel_id}_{sid}_resdiv{layer_div}_latentls{latent_layers}_{reweighted_id}_decay{weight_decay}"):
        print("This model exists")
        return
    

    ADNI_ds = MRI_3D_nolabels_inram_ds("/home/s2208943/ipdis/data/preprocessed_data/ADNI_300")
    IDs = ADNI_ds.getIDs()

    model_name = "SSN_Ens_Mean"
    output_maps_dir = f"/home/s2208943/ipdis/data/preprocessed_data/ADNI_300_output_maps/{model_name}/"
    output_maps, key_order = get_output_maps_for_ds(output_maps_dir, IDs)

    ### LOADING ADNI SPREADSHEETS
    adni_dir = "/home/s2208943/ipdis/data/ADNI_data/"
    spreadsheet_dir = adni_dir
    # dataset with clinical variables (e.g age, and a bunch of other factors)
    variables_df = pd.read_excel(spreadsheet_dir + "ADNI_300_Variables_for_Analysis.xlsx")

    # dataset with fazekas ratings by Maria. I only have 290 of the 298 due to a few images missing matches, which is a shame
    # but hopefully this is enough information. Nice.
    ratings_df = pd.read_excel(spreadsheet_dir + "ADNI_300_Sample_MVH_ratings.xlsx")

    

    combined_df = prepare_ADNI_dfs(ratings_df, variables_df)


    ### SETTING UP COMBINED DATASET AND DATALOADER
    fields = ['AGE', 'Ventricles_bl %', 'Hippocampus_bl %',
           'WholeBrain_bl %', 'Entorhinal_bl %', 'Fusiform_bl %', 'MidTemp_bl %',
           'BMI', 'PTGENDER', 'E-M RISK FACTORS',
           'DX.bl_0', 'DX.bl_1', 'DX.bl_2', 'DX.bl_3', 'CV RISK FACTORS_0',
           'CV RISK FACTORS_1', 'CV RISK FACTORS_2', 'APOE4_0', 'APOE4_1',
           'APOE4_2']
    imgs_ds = ADNI_ds
    imgs_ds = ExtractYChannelDataset(imgs_ds, 1)
    imgs_ds = AddChannelsDataset(imgs_ds, output_maps, IDs)
    imgs_ds = ImgsAndDfDataset(imgs_ds, combined_df, selected_fields=fields, label_field=label)
    imgs_ds = SkipBadIndexesDs(imgs_ds)
    imgs_ds = ConsumedDataset(imgs_ds)
    
    xent_class_weights = None
    if reweighted:
        labels = []
        for data in imgs_ds:
            labels.append(data[-1].item())
        labels = torch.Tensor(labels)
        class_counts = [(labels==i).sum().item() for i in range(4)]
        print(class_counts)
        xent_class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

    # used an online random number generator to get these
    combined_preds_val = []
    combined_labels_val = []
    combined_preds_test = []
    combined_labels_test = []
    for split in range(6):
        train_ds, val_ds, test_ds = cross_validate_split(imgs_ds, val_prop=0.15, test_prop=0.15, seed=3407, split=split, test_fold_smooth=1)
        # train_ds = AugmentedDataset(train_ds, get_transforms())
        # val_ds = AugmentedDataset(val_ds, get_transforms())
        test_ds = test_ds# AugmentedDataset(test_ds, get_transforms())

        # channels = ['pred']
        train_ds_2d = ConsumedDataset(FormatDataset(train_ds, channels=channels))
        val_ds_2d = ConsumedDataset(FormatDataset(val_ds, channels=channels))
        test_ds_2d = FormatDataset(test_ds, channels=channels)

        batch_size = 12
        train_dataloader = DataLoader(train_ds_2d, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_dataloader = DataLoader(val_ds_2d, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        test_dataloader = DataLoader(test_ds_2d, batch_size=batch_size, shuffle=False, num_workers=4)

        image_channels = train_ds_2d[0][0][0].shape[0]
        print(image_channels)
        clin_features = len(fields)
        torch.manual_seed(seed)
        # model_raw = PredictionModel(image_channels=image_channels, num_clin_features=clin_features, out_classes=4, channel_size_divide=8, finetune_head=True)#.cuda()
        model_raw = PredictionModelNHeadLayer(image_channels=image_channels, num_clin_features=clin_features, out_classes=4, channel_size_divide=layer_div, finetune_head=True, n_heads=latent_layers)
        loss = xent_wrapper(reweighted, xent_class_weights)


        ### training the model
        # setup optimizer and model wrapper

        # weight_decay = 0.01#0.05 #weight_decay
        max_epochs = 100
        lr=2e-4
        early_stop_patience = 7

        optimizer_params={"lr":lr, "weight_decay":weight_decay}
        optimizer = torch.optim.Adam
        lr_scheduler_params={"milestones":[1000], "gamma":0.5}
        lr_scheduler_constructor = torch.optim.lr_scheduler.MultiStepLR

        # wrap the model in the pytorch_lightning module that automates training
        # todo: this script needs filling in. I need to be able to replicate this for each model type which is why I have done this
        # I then need to do it for PVWMH as well...
        model = StandardLitModelWrapper(model=model_raw, loss=loss, 
                                        logging_metric=lambda : None,
                                        optimizer_params=optimizer_params,
                                        lr_scheduler_params=lr_scheduler_params,
                                        optimizer_constructor=optimizer,
                                        lr_scheduler_constructor=lr_scheduler_constructor,
                                       )

        # train the model
        trainer = get_trainer(max_epochs, f"./run{torch.randint(size=(1,), low=0,high=1000, generator=genner).item()}/", early_stop_patience=early_stop_patience)

        if not dummy_run:
            trainer.fit(model, train_dataloader, val_dataloader)

        r = trainer.validate(model, val_dataloader, ckpt_path='best')
        print(f"val_loss: {r[0]['val_loss']}")

        preds_val = []
        ys_val = []

        for data in val_dataloader:
            with torch.no_grad():
                (x, clin_data), y = data
                out = model.cuda()((x.cuda(), clin_data.cuda())).cpu()
                pred = torch.nn.functional.softmax(out, dim=1)
                preds_val.extend(pred.argmax(dim=1).cpu().numpy())
                ys_val.extend(y.cpu().numpy())


        preds_test = []
        ys_test = []

        for data in test_dataloader:
            with torch.no_grad():
                (x, clin_data), y = data
                out = model.cuda()((x.cuda(), clin_data.cuda())).cpu()
                pred = torch.nn.functional.softmax(out, dim=1)
                preds_test.extend(pred.argmax(dim=1).cpu().numpy())
                ys_test.extend(y.cpu().numpy())

        combined_preds_val.extend(preds_val)
        combined_preds_test.extend(preds_test)
        combined_labels_val.extend(ys_val)
        combined_labels_test.extend(ys_test)
        
        if dummy_run:
            break
            
    print("combined preds val")
    print(combined_preds_val)
    print("combined labels val")
    print(combined_labels_val)
    print("combined preds test")
    print(combined_preds_test)
    print("combined labels test")
    print(combined_labels_test)

    # DO WRITING OF THE RESULTS
    with open(f"/home/s2208943/ipdis/UQ_WMH_methods/trustworthai/run/ADNI_300_analysis/new_fazekas_prediction_code/results/ADNI/ADNI_{label}_{channel_id}_{sid}_resdiv{layer_div}_latentls{latent_layers}_{reweighted_id}_decay{weight_decay}", "w") as f:
        f.write(f"preds_val: {combined_preds_val}\n")
        f.write(f"labels_val: {combined_labels_val}\n")
        f.write(f"preds_test: {combined_preds_test}\n")
        f.write(f"labels_test: {combined_labels_test}\n")

if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
