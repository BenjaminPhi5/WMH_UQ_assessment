source ~/.bashrc
source activate wmh

# run on MSS3
python ventricle_extraction.py -i /home/s2208943/ipdis/data/preprocessed_data/MSS3_InterRaterData/imgs

# run on LBC
python ventricle_extraction.py -i /home/s2208943/ipdis/data/preprocessed_data/LBC_InterRaterData/imgs

# run on WMH Challenge inter rater version
python ventricle_extraction.py -i /home/s2208943/ipdis/data/preprocessed_data/WMHChallenge_InterRaterData/imgs
