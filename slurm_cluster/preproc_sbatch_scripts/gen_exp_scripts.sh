source /home/s2208943/miniconda3/etc/profile.d/conda.sh
conda activate ip

mv ~/ipdis/data/core_data/mixedCVDrelease/ ~/ipdis/data/core_data/EdData


python gen_preprocess_experiments.py -n EdData -d /home/${USER}/ipdis/data/core_data/ -p 20 -f True
python gen_preprocess_experiments.py -n WMH_challenge_dataset -d /home/${USER}/ipdis/data/extra_data/MRI_IP_project/ -p 8 -f True
