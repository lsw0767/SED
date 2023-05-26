#!/bin/sh
source ~/sed/venv/bin/activate
workspace=$workspace

export CUDA_VISIBLE_DEVICES=1


############# Plot figures for paper #############
python plot_for_paper.py mel_masks --workspace=$workspace --scene_type=dcase2018_task1 --snr=0 --holdout_fold=1 --cuda --mixture_type=original --model_type=thornnet --iteration=10000
#python plot_for_paper.py waveform --workspace=$workspace --scene_type=dcase2018_task1 --snr=0 --holdout_fold=1 --cuda --mixture_type=original
#python plot_for_paper.py mel_masks --workspace=$workspace --model_type=mhgwrp --scene_type=dcase2018_task1 --snr=0 --holdout_fold=1 --iteration=10000 --cuda
