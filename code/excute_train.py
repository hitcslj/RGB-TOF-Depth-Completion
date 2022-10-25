import os

command = "make train train_data_dir='/data/RGB_TOF/train' \
train_save_dir='/data/RGB_TOF/experiments_final_multiTest/' \
train_exp_id='baseline_wo_dilation'"

os.system(command)