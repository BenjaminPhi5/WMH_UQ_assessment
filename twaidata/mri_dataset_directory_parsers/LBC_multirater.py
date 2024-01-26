"""
directory parser for the LBC multi rater dataset
"""

from twaidata.mri_dataset_directory_parsers.generic import DirectoryParser
from twaidata.MRI_preprep.io import FORMAT
import os
import importlib.resources as pkg_resources
import twaidata.mri_dataset_directory_parsers.ed_domains_map as edm


class LBCMultiRaterDataParser(DirectoryParser):
    """
    structure of LBCMultiRaterDataParser:
    
    LBC1921_20930/
        LBC1921_20930_brainmask.nii.gz
        LBC1921_20930_FLAIRbrain.nii.gz
        LBC1921_20930_ICV.nii.gz
        LBC1921_20930_T1Wbrain.nii.gz
        LBC1921_20930_WMH2.nii.gz
        LBC1921_20930_WMH5.nii.gz
        LBC1921_20930_CSF.nii.gz
        LBC1921_20930_GM.nii.gz
        LBC1921_20930_NAWM.nii.gz
        LBC1921_20930_T2Wbrain.nii.gz
        LBC1921_20930_WMH4.nii.gz
    ...
    ...
    LBC1921_20947/
        ...
    """
    
    def __init__(self, dataset_root_in, *args, **kwargs):
        super().__init__(dataset_root_in, *args, **kwargs)
        
    
    def _build_dataset_table(self):
        
    
    
if __name__ == "__main__":
    print("testing")
    parser = LBCMultiRaterDataParser(
        # paths on the cluster for the in house data
        "/home/s2208943/ipdis/data/InterRater_data",
        "/home/s2208943/ipdis/data/preprocessed_data/LBC_InterRaterData"
    )
    
    iomap = parser.get_dataset_inout_map()
    for key, value in iomap.items():
        print("individual: ", key)
        print("individual map:", value)