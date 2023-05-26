#!/bin/sh
#source ~/sed/venv/bin/activate
workspace=$workspace
mixture_type="original"

# Create DCASE 2018 Task 2 cross-validation csv
#python utils/create_mixture_yaml.py create_dcase2018_task2_cross_validation_csv --dcase2018_task2_dataset_dir="/home/swlee/data/dcase2018/task2" --workspace=$workspace --mixture_type=$mixture_type

# Create mixture yaml file.
#python utils/create_mixture_yaml.py create_mixture_yaml --dcase2018_task1_dataset_dir="/home/swlee/data/dcase2018/task1" --dcase2018_task2_dataset_dir="/home/swlee/data/dcase2018/task2" --workspace=$workspace --mixture_type=$mixture_type

# Create mixed audios
#python utils/create_mixed_audio.py --dcase2018_task1_dataset_dir="/home/swlee/data/dcase2018/task1" --dcase2018_task2_dataset_dir="/home/swlee/data/dcase2018/task2" --workspace=$workspace --scene_type=dcase2018_task1 --snr=0 --mixture_type=$mixture_type

# Calculate features
#python utils/features.py logmel --workspace=$workspace --scene_type=dcase2018_task1 --snr=0 --mixture_type=$mixture_type



export CUDA_VISIBLE_DEVICES=0
# original rand_init no_clip mix
for mixture_type in original rand_init no_clip mix
do
    for type in tp tunet
#    for type in thornnet
    do
        for i in 1 2 3 4
        do
            python main_pytorch.py train --workspace=$workspace --model_type=$type --scene_type=dcase2018_task1 --snr=0 --holdout_fold=$i --cuda --mixture_type=$mixture_type
            python main_pytorch.py inference --workspace=$workspace --model_type=$type --scene_type=dcase2018_task1 --snr=0 --holdout_fold=$i --iteration=10000 --cuda --mixture_type=$mixture_type
            #python utils/get_avg_stats.py single_fold --workspace=$workspace --filename=main_pytorch --model_type=$type --scene_type=dcase2018_task1 --holdout_fold=$i --snr=0 --mixture_type=$mixture_type

            #python plot_for_paper.py mel_masks --workspace=$workspace --scene_type=dcase2018_task1 --snr=0 --holdout_fold=1 --cuda --mixture_type=$mixture_type --model_type=$type --iteration=10000
        done
#        python utils/get_avg_stats.py all_fold --workspace=$workspace --filename=main_pytorch --model_type=$type --scene_type=dcase2018_task1 --snr=0
    done
done


############# Plot figures for paper #############
#python plot_for_paper.py waveform --workspace=$workspace --scene_type=dcase2018_task1 --snr=0 --holdout_fold=1 --cuda
#python plot_for_paper.py mel_masks --workspace=$workspace --model_type=thornnet --scene_type=dcase2018_task1 --snr=0 --holdout_fold=1 --iteration=10000 --cuda --mixture_type=original