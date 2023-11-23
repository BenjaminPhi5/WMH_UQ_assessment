# copies the collated data for Ed and WMH onto the cluster folder

home_dir=/home/${USER}/ipdis/data/preprocessed_data/collated
ed_home_dir=${home_dir}/EdData
wmh_home_dir=${home_dir}/WMH_challenge_dataset
scratch_dir=/disk/scratch_big/${USER}/ipdis/preprep/out_data/collated
ed_scratch=${scratch_dir}/EdData
wmh_scratch=${scratch_dir}/WMH_challenge_dataset

mkdir -p ${scratch_dir}

echo Copying Ed Data
rsync --archive --update --compress --progress ${ed_home_dir}/ ${ed_scratch}

echo Copying WMH Challenge Data
rsync --archive --update --compress --progress ${wmh_home_dir}/ ${wmh_scratch}