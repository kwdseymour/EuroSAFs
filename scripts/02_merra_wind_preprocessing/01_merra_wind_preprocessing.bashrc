bsub -n 32 -W 4:00 -R "rusage[mem=4096]" -J \"S01\" -oo $HOME/SAFlogistics/results/01_merra_wind_preprocessing python $HOME/SAFlogistics/scripts/01_merra_wind_preprocessing/01_merra_wind_preprocessing.py -d $HOME/SAFlogistics -p 32

