from twaidata.mri_dataset_directory_parsers.WMHChallenge import WMHChallengeDirParser
from twaidata.mri_dataset_directory_parsers.EdData import EdDataParser
from twaidata.mri_dataset_directory_parsers.MSS3_multirater import MSS3MultiRaterDataParser
from twaidata.mri_dataset_directory_parsers.LBC_multirater import LBCMultiRaterDataParser
from twaidata.mri_dataset_directory_parsers.from_text_file import FromFileParser
import os

# TODO convert this method into an ENUM class.
def select_parser(dataset_name, dataset_location, preprocessed_location, csv_filepath=None, add_dataset_name_to_folder_path=True):
    """
    dataset_location: the parent folder of the dataset
    preprocessed_location: the parent folder of the preprocessed datasets
    """
    if add_dataset_name_to_folder_path:
        data_in_dir = os.path.join(dataset_location, dataset_name)
        data_out_dir = os.path.join(preprocessed_location, dataset_name)
    else:
        data_in_dir = dataset_location
        data_out_dir = preprocessed_location
    
    if dataset_name == "WMH_challenge_dataset":
        return WMHChallengeDirParser(data_in_dir,data_out_dir)
    elif dataset_name == "EdData" or dataset_name == "mixedCVDrelease":
        return EdDataParser(data_in_dir, data_out_dir)
    elif dataset_name == "MSS3":
        return MSS3MultiRaterDataParser(data_in_dir, data_out_dir)
    elif dataset_name == "LBC":
        return LBCMultiRaterDataParser(data_in_dir, data_out_dir)
    else:
        return FromFileParser(csv_filepath, dataset_location, preprocessed_location)