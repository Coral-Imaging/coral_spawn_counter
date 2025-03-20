    #!/bin/bash

    #PBS -N test_script
    #PBS -l ncpus=1
    #PBS -l ngpus=1
    #PBS -l gputype=A100
    #PBS -l mem=16gb
    #PBS -l walltime=6:00:00
    #PBS -m abe

    cd $PBS_O_WORKDIR
    
    source /home/tsaid/miniforge3/bin/activate cslics
    # conda activate /home/tsaid/miniforge3/envs/cslics
    # conda activate cslics

    python3 /home/tsaid/code/coral_spawn_counter/train/train_cslics_subsurface.py
    # python3 myscript.py

    conda deactivate