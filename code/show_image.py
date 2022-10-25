from audioop import mul
import os
from random import randrange
from re import L
from cv2 import matMulDeriv
import numpy as np
import cv2
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import torch


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

data_dir = '/home/liujian03/slow/multi/baseline_wo_dilation/result/iPhone_dynamic'
data_list = os.path.join(data_dir,'data.list')

data_dir2 = '/data/RGB_TOF/test_input/iPhone_dynamic'
data_list2 = os.path.join(data_dir2,'data.list')

output_path = '/home/liujian03/slow/multi/baseline_wo_dilation/showImage2'
if not os.path.exists(output_path):
    os.makedirs(output_path)

def visualize(name,img):
    plt.imshow(img,cmap='jet_r')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(os.path.join(output_path,name))
    plt.close()

sample_list = []
with open(data_list,'r') as f:
    lines = f.readlines()
    for l in lines:
        path = l.rstrip()
        sample_list.append(os.path.join(data_dir,path))

sample_list2 = []
with open(data_list2,'r') as f:
    lines = f.readlines()
    for l in lines:
        path = l.rstrip().split()[0]
        sample_list2.append(os.path.join(data_dir2,path))
for i in range(0,200,25):
    tmp = torch.from_numpy(cv2.imread(sample_list[i],cv2.IMREAD_ANYDEPTH))
    tmp2 = torch.from_numpy(cv2.imread(sample_list[i+1],cv2.IMREAD_ANYDEPTH))
    rgb = torch.from_numpy(cv2.imread(sample_list2[i])).permute(2,0,1)
    rgb2 = torch.from_numpy(cv2.imread(sample_list2[i+1])).permute(2,0,1)
    visualize('dep_pred'+str(i),tmp)
    visualize('rgb'+str(i),rgb[0,:,:])
    visualize('dep_pred'+str(i+1),tmp2)
    visualize('rgb'+str(i+1),rgb2[0,:,:])
    
   
    


  