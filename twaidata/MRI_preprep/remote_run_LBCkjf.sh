source ~/.bashrc
source activate wmh

in_dir=/home/s2208943/ipdis/data/InterRater_data
out_dir=/home/s2208943/ipdis/data/preprocessed_data/LBCkjf_InterRaterData

# process MSS3 dataset
python simple_preprocess_st2.py -i ${in_dir} -o ${out_dir} -n LBCkjf -s 0 -e -1 -f False -z True -a False -k False