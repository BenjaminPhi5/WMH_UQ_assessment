from twaidata.mri_dataset_directory_parsers.WMHChallenge import WMHChallengeDirParser
from twaidata.mri_dataset_directory_parsers.EdData import EdDataParser
from twaidata.mri_dataset_directory_parsers.from_text_file import FromFileParser
import os

# TODO convert this method into an ENUM class.
def select_parser(dataset_name, dataset_location, preprocessed_location, csv_filepath=None):
    """
    dataset_location: the parent folder of the dataset
    preprocessed_location: the parent folder of the preprocessed datasets
    """
    data_in_dir = os.path.join(dataset_location, dataset_name)
    data_out_dir = os.path.join(preprocessed_location, dataset_name)
    
    if dataset_name == "WMH_challenge_dataset":
        return WMHChallengeDirParser(data_in_dir,data_out_dir)
    elif dataset_name == "EdData" or dataset_name == "mixedCVDrelease":
        return EdDataParser(data_in_dir, data_out_dir)
    else:
        return FromFileParser(csv_filepath, dataset_location, preprocessed_location)