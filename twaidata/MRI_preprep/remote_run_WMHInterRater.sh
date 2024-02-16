source ~/.bashrc
source activate wmh

# note this is for the InterRater component of the WMH challenge dataset only.

in_dir=/home/s2208943/ipdis/data/WMH_Challenge_full_dataset
out_dir=/home/s2208943/ipdis/data/preprocessed_data/WMHChallenge_InterRaterData

# process MSS3 dataset
python simple_preprocess_st2.py -i ${in_dir} -o ${out_dir} -n WMH_InterRater -s 0 -e -1 -f False -z True -a False -k True