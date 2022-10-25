#!/usr/bin/env bash
train_data_dir='/data/RGB_TOF/data/train'
train_save_dir='/data/RGB_TOF/data/experiments/'
train_exp_id='01'
test_data_dir='/data/RGB_TOF/data/validation_input'
test_data_type='synthetic'
pretrain='/data/RGB_TOF/data/experiments/new_backbone/model_00100.pt'
test_save_dir='results'
 

# make train train_data_dir='/data/RGB_TOF/data/train' train_save_dir='/data/RGB_TOF/data/experiments_new/' train_exp_id='baseline_backbone _multi_bn_smooth'
train:
	rlaunch --gpu=8 --memory=200000 --cpu=16 -- \
	python3 -m torch.distributed.launch --nproc_per_node=8 main.py \
	--data_dir ${train_data_dir} \
	--data_list "data.list" \
	--save_dir ${train_save_dir} \
	--save ${train_exp_id} \
	--epochs 100 \
	--batch_size 1 \
	--lr 0.0001 \
	--decay 10,20,30,40,50 \
	--gamma 1.0,0.5,0.25,0.125,0.0625 \
	--max_depth 10.0 \
	--num_threads 16

# make test test_data_dir='/data/RGB_TOF/validation_input' \
test_data_type='synthetic' pretrain='/data/RGB_TOF/experiments_new/new_backbone/model_00050.pt' test_save_dir='results_' train_exp_id='new_backbone'
test:
	python3 -m torch.distributed.launch --nproc_per_node=1 main.py \
	--test_only \
	--max_depth 10.0 \
	--pretrain ${pretrain}

validate:
	python3 -m torch.distributed.launch --nproc_per_node=1 main.py \
	--validate_only \
	--max_depth 10.0 \
	--pretrain ${pretrain}