Running Sessions
----
Notes on running code on the internal cluster.

### Interactive Session
dont do processing on the head node.
find out which nodes are for interactive sessions

```
$ sinfo -o '%R;%N;%l' | column -s';' -t
> PARTITION          NODELIST                           TIMELIMIT
> Teach-Interactive  landonia[01,03,25]                 2:00:00
> …
> PGR-Standard       damnii[01-10]                      2:00:00
> …
```

`srun` can give us an interactive session on that head node: here is a job to run for 1 hour before timing out, using 1 cpu and 8GB of RAM
note that it uses no GPU (e.g for simple data exploration, or looking at results)

`srun --partition=PGR-Standard --time=01:00:00 --mem=8000 --cpus-per-task=1 --pty bash`

Always set a reasonble time limit like we do above so that if I forget to close a job after I am done it will just quit on me and I don't hog resources.

To request one GPU on my job, (say for debugging model code, check that a model will begin training without memory errors etc add the following flag (here for one GPU)):

`--gres=gpu:1`

or I can just run the commands from cluster scripts:

`$ interactive`

or 

`$ interactive-gpu`

which does something similar to the above (on an arbitrary node, with only 4gb ram and one 1cpu), less to remeber to do, nice.

**setting up a jupyter notebook**
after stating an srun job to login to a compute node and switching into conda env, start a jupyter notebook using:

`jupyter-lab --no-browser --ip=* --port=8081`

in another terminal in my home machine, forward a port to connect to the notebook server:

`ssh -L 8081:[COMPUTE_NODE_NAME]:8081 -fN s2208943@mlp.inf.ed.ac.uk`

paste the url from the first terminal window into browser.

To still be able to access the compute node while jupyter is running, use a tmux session to run the jupyter botebook from

### Batch Jobs and array jobs
the mnist tutorial for slurm is here:
https://github.com/cdt-data-science/cluster-scripts/tree/master/experiments/examples/mnist

follow this example, but the key is to setup a train.py file that takes a bunch of parameters, and does model setup and checkpointing etc
and a file for generating a bunch of experiments to try (a gen_experiments file) which creates an experiments.txt file
and a slurm array file that runs each script, copies all the data onto the scratch space and at the end deletes all the data from the scratch space,
and copies it back to the head node.

the slurm script has a bunch of parameters that i need to ensure i set up in a simple way, in particular the output folders both in scratch and in my project folder. Need somewhere to backup my different experiment data and log files (should use git, so those need to go somewhere useful). then i can just delete the output of runs that are not useul.

use squeue or whoson to see how many jobs i have pending. use killmyjobs to get rid of all jobs or scancel to get rid of a specific job (or an array of jobs). the batch array is really useful actually for easy parallelism as well, nice.
need to set it up to only select the pgr nodes so that i get the good gpus.

monitor slurm logs (which are updated periodically) for logs: `~/slum_logs` just contains the standard out so really useful.
