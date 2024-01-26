"""
directory parser for the MSS3 multi rater dataset
"""

from twaidata.mri_dataset_directory_parsers.generic import DirectoryParser
from twaidata.MRI_preprep.io import FORMAT
import os
import importlib.resources as pkg_resources
import twaidata.mri_dataset_directory_parsers.ed_domains_map as edm


class MSS3MultiRaterDataParser(DirectoryParser):
    """
    structure of MSS3MultiRaterDataParser:
    MSS3_ED_001/
        V1/
            MSS3_ED_001_V1_FLAIRbrain.nii.gz
            MSS3_ED_001_V1_lacune.nii.gz
            MSS3_ED_001_V1_T1Wbrain.nii.gz
            MSS3_ED_001_V1_T2Wbrain.nii.gz
            MSS3_ED_001_V1_WMH_mask_ES.nii.gz
            MSS3_ED_001_V1_WMH_mask_MVH.nii.gz
        V2/
            ...
    ...
    ...
    MSS3_ED_079/
        ...
    """
    
    def __init__(self, dataset_root_in, *args, **kwargs):
        super().__init__(dataset_root_in, *args, **kwargs)
        
    
    def _build_dataset_table(self):
        
    
    
if __name__ == "__main__":
    print("testing")
    parser = MSS3MultiRaterDataParser(
        # paths on the cluster for the in house data
        "/home/s2208943/ipdis/data/InterRater_data",
        "/home/s2208943/ipdis/data/preprocessed_data/MSS3_InterRaterData"
    )
    
    iomap = parser.get_dataset_inout_map()
    for key, value in iomap.items():
        print("individual: ", key)
        print("individual map:", value)