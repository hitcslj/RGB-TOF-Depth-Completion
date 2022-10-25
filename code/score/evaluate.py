import os
import cv2
import sys
import numpy as np
from utils import *
import matplotlib.pyplot as plt

frame_perscene = 25

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def evaluation(data_list_path, res_list_path, data_type):
    assert(data_type == 'synthetic' or data_type == 'iPhone_static' or data_type ==
           'iPhone_dynamic' or data_type == 'modified_phone_static')

    if data_type == 'modified_phone_static':
        depsp_list = read_datawoGT_list(data_list_path)
    else:
        depsp_list, gt_list = read_datawithGT_list(data_list_path)
    preddep_list = read_result_list(res_list_path)
    frame_num = len(preddep_list)

    if not data_type == 'modified_phone_static':
        rmae_all = np.zeros((frame_num,))
        ewmae_all = np.zeros((frame_num,))
        rds_all = np.zeros((frame_num,))
        for f, (depsp, gt, preddep) in enumerate(zip(depsp_list, gt_list, preddep_list)):
            rmae = RMAE(preddep, gt)
            ewmae = EWMAE(preddep, gt)
            rds = RDS(preddep, depsp)

            rmae_all[f] = rmae
            ewmae_all[f] = ewmae
            rds_all[f] = rds

    if data_type == 'iPhone_static' or data_type == 'modified_phone_static':
        scene_num = int(frame_num / frame_perscene)
        rtsd_all = np.zeros((scene_num,))
        for s in range(scene_num):
            start_idx = int(s*frame_perscene)
            end_idx = int((s+1)*frame_perscene)
            preddeps = np.stack(preddep_list[start_idx:end_idx], axis=0)
            rtsd = RTSD(preddeps)
            rtsd_all[s] = rtsd

    if data_type == 'synthetic':
        return rmae_all, ewmae_all, rds_all
    if data_type == 'iPhone_static':
        return rmae_all, ewmae_all, rds_all, rtsd_all
    if data_type == 'iPhone_dynamic':
        return rmae_all, ewmae_all, rds_all
    if data_type == 'modified_phone_static':
        return rtsd_all


def  test_score():
    # path to the validate result
    submit_dir = '/home/liujian03/slow/multi/baseline_wo_dilation/validate'
    # path to the validation ground truth
    truth_dir = '/home/liujian03/slow/multi/baseline_wo_dilation/score/validation_reference'
    output_filename =  '/home/liujian03/slow/multi/baseline_wo_dilation/score/score.txt'

    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)
    # Calculate metrics
    rmae_total, ewmae_total, rds_total, rtsd_total = [], [], [], []
    for data_type in ['synthetic', 'iPhone_static', 'iPhone_dynamic', 'modified_phone_static']:
        print(f'Evaluating {data_type} data')
        data_list_path = os.path.join(truth_dir, data_type, 'data.list')
        print(data_list_path)
        res_list_path = os.path.join(submit_dir, data_type, 'data.list')

        if data_type == 'synthetic':
            rmae_all, ewmae_all, rds_all = evaluation(
                data_list_path, res_list_path, data_type)
            rmae_total = np.concatenate((rmae_total, rmae_all))
            ewmae_total = np.concatenate((ewmae_total, ewmae_all))
            rds_total = np.concatenate((rds_total, rds_all))
        if data_type == 'iPhone_static':
            rmae_all, ewmae_all, rds_all, rtsd_all = evaluation(
                data_list_path, res_list_path, data_type)
            rmae_total = np.concatenate((rmae_total, rmae_all))
            ewmae_total = np.concatenate((ewmae_total, ewmae_all))
            rds_total = np.concatenate((rds_total, rds_all))
            rtsd_total = np.concatenate((rtsd_total, rtsd_all))
        if data_type == 'iPhone_dynamic':
            rmae_all, ewmae_all, rds_all = evaluation(
                data_list_path, res_list_path, data_type)
            rmae_total = np.concatenate((rmae_total, rmae_all))
            ewmae_total = np.concatenate((ewmae_total, ewmae_all))
            rds_total = np.concatenate((rds_total, rds_all))
        if data_type == 'modified_phone_static':
            rtsd_all = evaluation(data_list_path, res_list_path, data_type)
            rtsd_total = np.concatenate((rtsd_total, rtsd_all))

    # Calculate final scores
    metric_vals = [rmae_total, ewmae_total, rds_total, rtsd_total]
    metric_names = ['RMAE', 'EWMAE', 'RDS', 'RTSD']
    metric_weights = [1.8, 0.6, 3, 4.6]

    # Write the result into score_path/score.txt
    total_score = 0
    with open(output_filename, 'w') as f:
        for metric_val, metric_weight, metric_name in zip(metric_vals, metric_weights, metric_names):
            metric_mean = np.mean(metric_val)
            f.write(f'{metric_name}: {metric_mean:.3f}\n')
            print(f'{metric_name}: {metric_mean:.3f}\n')
            total_score += metric_weight * np.mean(metric_mean)
        total_score = 1 - total_score
        f.write(f'SCORE: {total_score:.3f}\n')
        print(f'SCORE: {total_score:.3f}\n')

if __name__ == '__main__':
    test_score()