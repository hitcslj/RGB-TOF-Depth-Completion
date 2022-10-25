import os
import cv2
import numpy as np
import sys

### Read depth data
def read_depth_data(dep_path):
    if dep_path.split('.')[-1] == 'bin':
        dep = np.fromfile(dep_path, np.float32)
        dep = dep.reshape(192,256)
        return dep
    elif dep_path.split('.')[-1] == 'exr':
        dep = cv2.imread(dep_path, cv2.IMREAD_ANYDEPTH)
        return dep
    else:
        sys.exit('Cannot read depth data!')
        
def read_datawithGT_list(data_list_path):
    data_dir = data_list_path.split('/' + data_list_path.split('/')[-1])[0]
    depsp_list = []
    gt_list = []
    with open(data_list_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            paths = l.rstrip().split()
            depsp_list.append(read_depth_data(os.path.join(data_dir, paths[0])))
            gt_list.append(read_depth_data(os.path.join(data_dir, paths[1])))
    return depsp_list, gt_list

def read_datawoGT_list(data_list_path):
    data_dir = data_list_path.split('/' + data_list_path.split('/')[-1])[0]
    depsp_list = []
    with open(data_list_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            paths = l.rstrip().split()
            depsp_list.append(read_depth_data(os.path.join(data_dir, paths[0])))
    return depsp_list

def read_result_list(data_list_path):
    data_dir = data_list_path.split('/' + data_list_path.split('/')[-1])[0]
    preddep_list = []
    with open(data_list_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            paths = l.rstrip().split()
            preddep_list.append(read_depth_data(os.path.join(data_dir, paths[0])))
    return preddep_list


# Relative Mean Absolute Error
def RMAE(pred_dep, gt):
    rmae = np.mean(np.abs((pred_dep[gt>0]-gt[gt>0])/gt[gt>0]))
    return rmae


# Edge Weighted Mean Absolute Error
def EWMAE(work_image, ref_image, kappa=0.5):
    """GCMSE --- Gradient Conduction Mean Square Error.

    Computation of the GCMSE. An image quality assessment measurement 
    for image filtering, focused on edge preservation evaluation. 

    gcmse: float
        Value of the GCMSE metric between the 2 provided images. It gets
        smaller as the images are more similar.
    """
    # Normalization of the images to [0,1] values.
    max_val = ref_image.max()
    ref_image_float = ref_image.astype('float32')
    work_image_float = work_image.astype('float32')	
    normed_ref_image = ref_image_float / max_val
    normed_work_image = work_image_float / max_val
    
    # Initialization and calculation of south and east gradients arrays.
    gradient_S = np.zeros_like(normed_ref_image)
    gradient_E = gradient_S.copy()
    gradient_S[:-1,: ] = np.diff(normed_ref_image, axis=0)
    gradient_E[: ,:-1] = np.diff(normed_ref_image, axis=1)
    
    # Image conduction is calculated using the Perona-Malik equations.
    cond_S = np.exp(-(gradient_S/kappa) ** 2)
    cond_E = np.exp(-(gradient_E/kappa) ** 2)
        
    # New conduction components are initialized to 1 in order to treat
    # image corners as homogeneous regions
    cond_N = np.ones_like(normed_ref_image)
    cond_W = cond_N.copy()
    # South and East arrays values are moved one position in order to
    # obtain North and West values, respectively. 
    cond_N[1:, :] = cond_S[:-1, :]
    cond_W[:, 1:] = cond_E[:, :-1]
    
    # Conduction module is the mean of the 4 directional values.
    conduction = (cond_N + cond_S + cond_W + cond_E) / 4
    conduction = np.clip (conduction, 0., 1.)
    G = 1 - conduction
    
    # Calculation of the GCMAE value 
    ewmae = (abs(G*(normed_ref_image - normed_work_image))).sum()/ G.sum()
    return ewmae

# Relative Depth shift
def RDS(pred_dep, depsp):
    x_sp, y_sp = np.where(depsp>0)
    d_sp = depsp[x_sp, y_sp]
    rds = np.mean(abs(pred_dep[x_sp, y_sp]-d_sp)/d_sp)
    return rds

# Relative Temporal Standard Deviation
def RTSD(pred_deps):
    rtsd = np.mean(np.std(pred_deps, axis=0)/np.mean(pred_deps,axis=0))
    return rtsd