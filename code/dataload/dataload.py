import os
import random
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from . import BaseDataLoad

def ReadData(data_list):
    sample_list = []
    with open(data_list, 'r') as f:
        lines = f.readlines()
        for l in lines:
            paths = l.rstrip().split()
            sample_list.append(paths)
    return sample_list

class DataLoad(BaseDataLoad):
    def __init__(self, args, mode,data_type=None,validate=False):
        super(DataLoad, self).__init__(args, mode)
        self.args = args
        self.mode = mode
        assert(mode == 'train' or mode=='val' or mode=='test')
        if mode=='val':
            data_dir1 = '/data/RGB_TOF/validation_input/all'
            data_dir2 = '/data/RGB_TOF/validation_reference/all'
            data_list1 = os.path.join(data_dir1, 'data.list')
            data_list2 = os.path.join(data_dir2, 'data.list')
            validation_input = ReadData(data_list1)
            validation_reference = ReadData(data_list2)
            n = len(validation_input)
            self.sample_list = []
            for i in range(n):
                self.sample_list.append({'rgb':os.path.join(data_dir1,validation_input[i][0]),'depth':os.path.join(data_dir1,validation_input[i][1]),'gt':os.path.join(data_dir2,validation_reference[i][1])})
        elif mode=='train':
            self.scale =  (192,256)
            self.augment = self.args.augment
            self.sample_list = []
            scenes = dict()
            data_dir = self.args.data_dir
            data_list = os.path.join(data_dir, 'data.list')
            with open(data_list, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    paths = l.rstrip().split()
                    scene = paths[0].split('/')[0]
                    if scene in scenes:
                        scenes[scene] += 1
                    else:
                        scenes[scene] = 1
                    self.sample_list.append({'rgb': os.path.join(data_dir, paths[0]), 'depth': os.path.join(data_dir, paths[1])})
            n = len(self.sample_list)
            tmp = []
            total = 0 
            for scene in scenes:
                num_pic = scenes[scene]
                for j in range(num_pic-24):
                    frames = []
                    depth_frames = []
                    for k in range(25):
                        frames.append(self.sample_list[total+j+k]['rgb'])
                        depth_frames.append(self.sample_list[total+j+k]['depth'])
                    tmp.append({"rgb":frames, "depth":depth_frames})
                total +=num_pic
            self.sample_list = tmp
        elif mode=='test':
            if not validate:data_dir = '/data/RGB_TOF/test_input'
            else:data_dir = '/data/RGB_TOF/validation_input'
            data_dir = os.path.join(data_dir,data_type)
            data_list = os.path.join(data_dir,'data.list')
            test_input = ReadData(data_list)
            n = len(test_input)
            self.sample_list = []
            for i in range(n):
                self.sample_list .append({"rgb":os.path.join(data_dir,test_input[i][0]), "depth":os.path.join(data_dir,test_input[i][1])})
    def __len__(self):
        return len(self.sample_list)
    def __getitem__(self, idx):
        if self.mode=='train':
            rgb = cv2.imread(self.sample_list[idx]['rgb'][0])
            dep = cv2.imread(self.sample_list[idx]['depth'][0], cv2.IMREAD_ANYDEPTH)
            dep = dep.astype(np.float32)
            # print(self.sample_list[idx]['depth'][0],dep.shape)
            dep[dep>self.args.max_depth] = 0
            dep = Image.fromarray(dep)
            rgb = Image.fromarray(rgb)
            t_dep = T.Compose([
                T.Resize(self.scale, T.InterpolationMode('nearest')),
                T.ToTensor()
            ])
            t_rgb = T.Compose([
                T.Resize(self.scale, T.InterpolationMode('nearest')),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            rgb = t_rgb(rgb)
            rgb_multi = torch.zeros((25, rgb.shape[0], rgb.shape[1], rgb.shape[2]))
            for i in range(25):
                rgb = cv2.imread(self.sample_list[idx]['rgb'][i])
                rgb = Image.fromarray(rgb)
                rgb = t_rgb(rgb)
                rgb_multi[i] = rgb
            dep = t_dep(dep)
            dep_multi = torch.zeros((25, dep.shape[0], dep.shape[1], dep.shape[2]))
            for i in range(25):
                dep = cv2.imread(self.sample_list[idx]['depth'][i],cv2.IMREAD_ANYDEPTH)
                dep[dep>self.args.max_depth] = 0 
                dep = Image.fromarray(dep)
                dep = t_dep(dep)
                # print(dep.shape)
                dep_multi[i]=dep
            dep_sp = self.get_sparse_depth_grid(dep_multi)
            if self.args.cut_mask:
                dep_sp = self.cut_mask(dep_sp)
            rgb_multi = rgb_multi.view(-1, rgb_multi.shape[-2], rgb_multi.shape[-1])
            dep_sp = dep_sp.view(-1, dep_sp.shape[-2], dep_sp.shape[-1])
            dep_gt = dep_multi.clone()
            dep_gt = dep_gt.view(-1, dep_gt.shape[-2], dep_gt.shape[-1])
            output = {'rgb': rgb_multi, 'dep': dep_sp, 'gt': dep_gt}
            return output

        elif self.mode=='val':
            rgb = cv2.imread(self.sample_list[idx]['rgb'])
            dep = cv2.imread(self.sample_list[idx]['depth'], cv2.IMREAD_ANYDEPTH)
            gt = cv2.imread(self.sample_list[idx]['gt'], cv2.IMREAD_ANYDEPTH)
            dep = dep.astype(np.float32)
            dep[dep>self.args.max_depth] = 0
            gt = gt.astype(np.float32)
            gt[gt>self.args.max_depth] = 0
            dep = Image.fromarray(dep)
            rgb = Image.fromarray(rgb)
            gt = Image.fromarray(gt)
            t_dep = T.Compose([
                T.ToTensor()
            ])
            t_rgb = T.Compose([
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            rgb = t_rgb(rgb)
            dep = t_dep(dep)
            gt = t_dep(gt)
            output = {'rgb': rgb, 'dep': dep,'gt':gt}
            return output
        elif self.mode=='test':
            rgb = cv2.imread(self.sample_list[idx]['rgb'])
            dep = cv2.imread(self.sample_list[idx]['depth'], cv2.IMREAD_ANYDEPTH)
            dep = dep.astype(np.float32)
            dep[dep>self.args.max_depth] = 0
            dep = Image.fromarray(dep)
            rgb = Image.fromarray(rgb)
            t_dep = T.Compose([
                T.ToTensor()
            ])
            t_rgb = T.Compose([
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            rgb = t_rgb(rgb)
            dep = t_dep(dep)
            output = {'rgb': rgb, 'dep': dep}
            return output


    def get_sparse_depth(self, dep, num_spot):
        dep_sp = torch.zeros_like(dep)
        for i in range(dep.shape[0]):
            _,channel, height, width = dep[i].shape
            assert channel == 1
            idx_nnz = torch.nonzero(dep[i].view(-1) > 0.0001)
            num_idx = len(idx_nnz)
            if num_idx == 0:
                continue
            idx_sample = torch.randperm(num_idx)[:num_spot]
            idx_nnz = idx_nnz[idx_sample[:]]
            mask = torch.zeros((channel*height*width))
            mask[idx_nnz] = 1.0
            mask = mask.view((channel, height, width))
            dep_sp[i] = dep[i] * mask.type_as(dep[i])
        return dep_sp

    def get_sparse_depth_grid(self, dep):
        '''
        Simulate pincushion distortion:
        --stride: 
        It controls the distance between neighbor spots7
        Suggest stride value:       5~10
        --dist_coef:
        It controls the curvature of the spot pattern
        Larger dist_coef distorts the pattern more.
        Suggest dist_coef value:    0 ~ 5e-5
        --noise:
        standard deviation of the spot shift
        Suggest noise value:        0 ~ 0.5
        '''
        # Generate Grid points
        dep_sp = torch.zeros_like(dep)
        for i in range(dep.shape[0]):
            channel, img_h, img_w = dep[i].shape
            assert channel == 1
            stride = np.random.randint(5,7)
            dist_coef = np.random.rand()*4e-5 + 1e-5
            noise = np.random.rand() * 0.3
        
            x_odd, y_odd = np.meshgrid(np.arange(stride//2, img_h, stride*2), np.arange(stride//2, img_w, stride))
            x_even, y_even = np.meshgrid(np.arange(stride//2+stride, img_h, stride*2), np.arange(stride, img_w, stride))
            x_u = np.concatenate((x_odd.ravel(),x_even.ravel()))
            y_u = np.concatenate((y_odd.ravel(),y_even.ravel()))
            x_c = img_h//2 + np.random.rand()*50-25
            y_c = img_w//2 + np.random.rand()*50-25
            x_u = x_u - x_c
            y_u = y_u - y_c       
        
            # Distortion
            r_u = np.sqrt(x_u**2+y_u**2)
            r_d = r_u + dist_coef * r_u**3
            num_d = r_d.size
            sin_theta = x_u/r_u
            cos_theta = y_u/r_u
            x_d = np.round(r_d * sin_theta + x_c + np.random.normal(0, noise, num_d))
            y_d = np.round(r_d * cos_theta + y_c + np.random.normal(0, noise, num_d))
            idx_mask = (x_d<img_h) & (x_d>0) & (y_d<img_w) & (y_d>0)
            x_d = x_d[idx_mask].astype('int')
            y_d = y_d[idx_mask].astype('int')
            spot_mask = np.zeros((img_h, img_w))
            spot_mask[x_d,y_d] = 1
            dep_sp[i,:, x_d, y_d] = dep[i,:, x_d, y_d]
        return dep_sp
    def cut_mask(self, dep):
        for i in range(dep.shape[0]):
            _, h, w = dep[i].size()
            c_x = np.random.randint(h/4, h/4*3)
            c_y = np.random.randint(w/4, w/4*3)
            r_x = np.random.randint(h/4, h/4*3)
            r_y = np.random.randint(h/4, h/4*3)
            mask = torch.zeros_like(dep[i])
            min_x = max(c_x-r_x, 0)
            max_x = min(c_x+r_x, h)
            min_y = max(c_y-r_y, 0)
            max_y = min(c_y+r_y, w)
            mask[0, min_x:max_x, min_y:max_y] = 1
            dep[i] = dep[i]*mask
        return dep