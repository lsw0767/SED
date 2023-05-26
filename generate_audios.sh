#!/bin/sh
#source ~/sed/venv/bin/activate
workspace=$workspace


for mixture_type in mix
do
#  # Create DCASE 2018 Task 2 cross-validation csv
#  python utils/create_mixture_yaml.py create_dcase2018_task2_cross_validation_csv --dcase2018_task2_dataset_dir="/home/swlee/data/dcase2018/task2" --workspace=$workspace --mixture_type=$mixture_type
#
#  # Create mixture yaml file.
#  python utils/create_mixture_yaml.py create_mixture_yaml --dcase2018_task1_dataset_dir="/home/swlee/data/dcase2018/task1" --dcase2018_task2_dataset_dir="/home/swlee/data/dcase2018/task2" --workspace=$workspace --mixture_type=$mixture_type
#
#  # Create mixed audios
#  python utils/create_mixed_audio.py --dcase2018_task1_dataset_dir="/home/swlee/data/dcase2018/task1" --dcase2018_task2_dataset_dir="/home/swlee/data/dcase2018/task2" --workspace=$workspace --scene_type=dcase2018_task1 --snr=0 --mixture_type=$mixture_type

  # Calculate features
  python utils/features.py logmel --workspace=$workspace --scene_type=dcase2018_task1 --snr=0 --mixture_type=$mixture_type

done
