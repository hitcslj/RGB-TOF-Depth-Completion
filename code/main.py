from locale import normalize
import os
import time
import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from dataload.dataload import DataLoad
from loss.loss import Loss
from config import get_parser
from model.res_backbone import UNet
import utility
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
def train(args,gpu=0):
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(args.local_rank)
    torch.cuda.set_device(local_rank)
    gpu = local_rank
    ### Load data
    data_train = DataLoad(args, 'train')
    data_val = DataLoad(args, 'val')
    sampler_train = DistributedSampler(
        data_train)
    batch_size = args.batch_size
    loader_train = DataLoader(
        dataset=data_train, batch_size=batch_size, shuffle=False,
        num_workers=args.num_threads, pin_memory=False,sampler=sampler_train,
        drop_last=True)
    loader_val = DataLoader(
        dataset=data_val, batch_size=25, shuffle=False,
        num_workers=args.num_threads, pin_memory=False,
        drop_last=False)
    
    ### Set up Network
    net = UNet(in_planes=4)
    net = net.cuda()
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device= local_rank, find_unused_parameters=False)
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)
        checkpoint = torch.load(args.pretrain)
        if gpu==0:
            net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()}, strict=False)
            print('Load network parameters from : {}'.format(args.pretrain))
    ### Set up Loss
    loss = Loss(args)
    ### Set up Optimizer
    optimizer, scheduler = utility.make_optimizer_scheduler(args, net)
    if gpu == 0:
        #utility.backup_source_code(args.save_dir + '/code')
        try:
            os.makedirs(args.save_dir, exist_ok=True)
            os.makedirs(args.save_dir + '/train', exist_ok=True)
            os.makedirs(args.save_dir + '/val', exist_ok=True)
 
        except OSError:
            pass
    ### Training
    for epoch in range(args.epochs+1):
        net.train()
        sampler_train.set_epoch(epoch)
        num_sample = len(loader_train) * loader_train.batch_size
        if gpu == 0:
            pbar = tqdm(total=num_sample*25)
            log_cnt = 0.0
            log_loss = 0.0
        for batch, sample in enumerate(loader_train):
            sample = {key: val.cuda(gpu) for key, val in sample.items() if val is not None}
            sample["rgb"] = sample["rgb"].view(-1, 3, sample["rgb"].shape[-2], sample["rgb"].shape[-1])
            sample["dep"] = sample["dep"].view(-1, 1, sample["dep"].shape[-2], sample["dep"].shape[-1])
            sample["gt"] = sample["gt"].view(-1, 1, sample["gt"].shape[-2], sample["gt"].shape[-1])
            output = net(sample)
            output[sample['dep']>0] = sample['dep'][sample['dep']>0]
            optimizer.zero_grad()
            loss_sum, loss_val = loss(sample, output)
            # Divide by batch size
            loss_sum = loss_sum / (loader_train.batch_size*25) 
            loss_val = loss_val / (loader_train.batch_size*25)
            loss_rmae = _RMAE(output, sample['gt'])      
            loss_rtsd = _RTSD(torch.squeeze(output), dim=0)
            loss_total = loss_sum + loss_rmae + 2*loss_rtsd 
            loss_total.backward()
            optimizer.step()
            if gpu == 0:
                log_cnt += 1
                log_loss += loss_total.item()
                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss1 = {:.4f} | RMAE = {:.4f} | rtsd={:.4f}'.format(
                    'Train', current_time, log_loss / log_cnt,loss_rmae,loss_rtsd)
                pbar.set_description(error_str)
                pbar.update(loader_train.batch_size*25)
        if gpu == 0:
            pbar.close()
            if args.save_full or epoch == args.epochs:
                state = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'args': args
                }
            else:
                state = {
                    'net': net.state_dict(),
                    'args': args
                }
            torch.save(state, f'{args.save_dir}/model_{epoch:05d}.pt')
        while(not os.path.exists('{}/model_{:05d}.pt'.format(args.save_dir, epoch))):
            #print('waiting for model')
            pass
        if gpu==0:
            checkpoint = torch.load('{}/model_{:05d}.pt'.format(args.save_dir, epoch))
            net.load_state_dict(checkpoint['net'], strict=False)
        
        ### Validation
        if (epoch+1) % 1 == 0:
            def test_score():
                num_sample = len(loader_val) * loader_val.batch_size
                pbar = tqdm(total=num_sample)
                # 0:iphone_dynamic 1:iphone_static 2:modified_iphone_static 3:sythetic
                data_types = ['iPhone_dynamic','iPhone_static','modified_phone_static','synthetic']
                rmae_total, ewmae_total, rds_total, rtsd_total = [], [], [], []
                for batch, sample in enumerate(loader_val):
                    sample = {key: val.cuda(gpu) for key, val in sample.items() if val is not None}
                    sample["rgb"] = sample["rgb"].view(-1, 3, sample["rgb"].shape[-2], sample["rgb"].shape[-1])
                    sample["dep"] = sample["dep"].view(-1, 1, sample["dep"].shape[-2], sample["dep"].shape[-1])
                    output = net(sample)
                    output[sample['dep']>0] = sample['dep'][sample['dep']>0]
                    data_type = data_types[batch]
                    sample['dep'] =  sample['dep'].detach().cpu().numpy()
                    sample['gt'] = sample['gt'].detach().cpu().numpy()
                    output = output.detach().cpu().numpy()
                    if not data_type == 'modified_phone_static':
                        rmae_all = np.zeros((25,))
                        ewmae_all = np.zeros((25,))
                        rds_all = np.zeros((25,))
                        for i in range(25):
                            preddep,depsp,gt = output[i][0],sample['dep'][i][0],sample['gt'][i][0]
                            rmae = RMAE(preddep, gt)
                            ewmae = EWMAE(preddep, gt)
                            rds = RDS(preddep, depsp)
                            rmae_all[i] = rmae
                            ewmae_all[i] = ewmae
                            rds_all[i] = rds
                    if data_type == 'iPhone_static' or data_type == 'modified_phone_static':
                        rtsd_all = np.zeros((1,))
                        preddeps =  np.squeeze(output)
                        rtsd = RTSD(preddeps)
                        rtsd_all[0] = rtsd
                    if data_type == 'synthetic':
                        rmae_total = np.concatenate((rmae_total, rmae_all))
                        ewmae_total = np.concatenate((ewmae_total, ewmae_all))
                        rds_total = np.concatenate((rds_total, rds_all))
                    if data_type == 'iPhone_static':
                        rmae_total = np.concatenate((rmae_total, rmae_all))
                        ewmae_total = np.concatenate((ewmae_total, ewmae_all))
                        rds_total = np.concatenate((rds_total, rds_all))
                        rtsd_total = np.concatenate((rtsd_total, rtsd_all))
                    if data_type == 'iPhone_dynamic':
                        rmae_total = np.concatenate((rmae_total, rmae_all))
                        ewmae_total = np.concatenate((ewmae_total, ewmae_all))
                        rds_total = np.concatenate((rds_total, rds_all))
                    if data_type == 'modified_phone_static':
                        rtsd_total = np.concatenate((rtsd_total, rtsd_all))
                    current_time = time.strftime('%y%m%d@%H:%M:%S')
                    score_str = 'name{:<10s}{:<10s}| {} | epoch {} '.format(
                    data_type,'Test', current_time,  epoch)
                    pbar.set_description(score_str)
                    pbar.update(loader_val.batch_size)
                    
                # Calculate final scores
                metric_vals = [rmae_total, ewmae_total, rds_total, rtsd_total]
                metric_names = ['RMAE', 'EWMAE', 'RDS', 'RTSD']
                metric_weights = [1.8, 0.6, 3, 4.6]

                # Write the result into score_path/score.txt
                total_score = 0
                with open(os.path.join(args.save_dir,'score.txt'), 'a') as f:
                    f.write(f'epoch:{epoch+1}\t')
                    print(f'epoch:{epoch+1}\t')
                    for metric_val, metric_weight, metric_name in zip(metric_vals, metric_weights, metric_names):
                        metric_mean = np.mean(metric_val)
                        f.write(f'{metric_name}: {metric_mean:.3f}\t')
                        print(f'{metric_name}: {metric_mean:.3f}\t')
                        total_score += metric_weight * np.mean(metric_mean)
                    total_score = 1 - total_score
                    f.write(f'SCORE: {total_score:.3f}\n')
                    print(f'SCORE: {total_score:.3f}\n') 
                pbar.close()

            def test_score_2():
                num_sample = len(loader_val) * loader_val.batch_size
                pbar = tqdm(total=num_sample)
                # 0:iphone_dynamic 1:iphone_static 2:modified_iphone_static 3:sythetic
                data_types = ['iPhone_dynamic','iPhone_static','modified_phone_static','synthetic']
                rmae_total, ewmae_total, rds_total, rtsd_total = [], [], [], []
                for batch, sample in enumerate(loader_val):
                    sample = {key: val.cuda(gpu) for key, val in sample.items() if val is not None}
                    sample["rgb"] = sample["rgb"].view(-1, 3, sample["rgb"].shape[-2], sample["rgb"].shape[-1])
                    sample["dep"] = sample["dep"].view(-1, 1, sample["dep"].shape[-2], sample["dep"].shape[-1])
                    output = net(sample)
                    output[sample['dep']>0] = sample['dep'][sample['dep']>0]
                    data_type = data_types[batch]
                    sample['dep'] =  sample['dep'].detach().cpu().numpy()
                    sample['gt'] = sample['gt'].detach().cpu().numpy()
                    output = output.detach().cpu().numpy()
                    if not data_type == 'modified_phone_static' :
                        rmae_all = np.zeros((25,))
                        ewmae_all = np.zeros((25,))
                        rds_all = np.zeros((25,))
                        if data_type=='iPhone_static':
                            for i in range(25):
                                preddep,depsp,gt = output[12][0],sample['dep'][i][0],sample['gt'][i][0]
                                rmae = RMAE(preddep, gt)
                                ewmae = EWMAE(preddep, gt)
                                rds = RDS(preddep, depsp)
                                rmae_all[i] = rmae
                                ewmae_all[i] = ewmae
                                rds_all[i] = rds
                        else:
                            for i in range(25):
                                preddep,depsp,gt = output[i][0],sample['dep'][i][0],sample['gt'][i][0]
                                rmae = RMAE(preddep, gt)
                                ewmae = EWMAE(preddep, gt)
                                rds = RDS(preddep, depsp)
                                rmae_all[i] = rmae
                                ewmae_all[i] = ewmae
                                rds_all[i] = rds
                    rtsd_all = np.array([0])
                    if data_type == 'synthetic':
                        rmae_total = np.concatenate((rmae_total, rmae_all))
                        ewmae_total = np.concatenate((ewmae_total, ewmae_all))
                        rds_total = np.concatenate((rds_total, rds_all))
                    if data_type == 'iPhone_static':
                        rmae_total = np.concatenate((rmae_total, rmae_all))
                        ewmae_total = np.concatenate((ewmae_total, ewmae_all))
                        rds_total = np.concatenate((rds_total, rds_all))
                        rtsd_total = np.concatenate((rtsd_total, rtsd_all))
                    if data_type == 'iPhone_dynamic':
                        rmae_total = np.concatenate((rmae_total, rmae_all))
                        ewmae_total = np.concatenate((ewmae_total, ewmae_all))
                        rds_total = np.concatenate((rds_total, rds_all))
                    if data_type == 'modified_phone_static':
                        rtsd_total = np.concatenate((rtsd_total, rtsd_all))
                    current_time = time.strftime('%y%m%d@%H:%M:%S')
                    score_str = 'name{:<10s}{:<10s}| {} | epoch {} '.format(
                    data_type,'Test', current_time,  epoch)
                    pbar.set_description(score_str)
                    pbar.update(loader_val.batch_size)
                    
                # Calculate final scores
                metric_vals = [rmae_total, ewmae_total, rds_total, rtsd_total]
                metric_names = ['RMAE', 'EWMAE', 'RDS', 'RTSD']
                metric_weights = [1.8, 0.6, 3, 4.6]

                # Write the result into score_path/score.txt
                total_score = 0
                with open(os.path.join(args.save_dir,'score_2.txt'), 'a') as f:
                    f.write(f'epoch:{epoch+1}\t')
                    print(f'epoch:{epoch+1}\t')
                    for metric_val, metric_weight, metric_name in zip(metric_vals, metric_weights, metric_names):
                        metric_mean = np.mean(metric_val)
                        f.write(f'{metric_name}: {metric_mean:.3f}\t')
                        print(f'{metric_name}: {metric_mean:.3f}\t')
                        total_score += metric_weight * np.mean(metric_mean)
                    total_score = 1 - total_score
                    f.write(f'SCORE_2: {total_score:.3f}\n')
                    print(f'SCORE_2: {total_score:.3f}\n') 
                pbar.close()
            
            torch.set_grad_enabled(False)
            net.eval()
            if gpu==0:
                test_score()
                test_score_2()
            torch.set_grad_enabled(True)
        scheduler.step()

def test(args):
    ### Prepare dataset
    data_types = ['iPhone_dynamic','iPhone_static','modified_phone_static','synthetic']
    for data_dype in data_types:
        data_test = DataLoad(args, 'test',data_dype)
        loader_test = DataLoader(dataset=data_test, batch_size=1, shuffle=False)
        ###  Set up Network
        net = UNet(in_planes=4)
        net.cuda()
        if args.pretrain is not None:
            assert os.path.exists(args.pretrain), \
                "file not found: {}".format(args.pretrain)

            checkpoint = torch.load(args.pretrain)
            net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()},strict=False)
            # net.load_state_dict(checkpoint['net'], strict=True)
        net.eval()

        num_sample = len(loader_test)*loader_test.batch_size

        output_path = os.path.join( './result',data_dype)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        pbar = tqdm(total=num_sample)
        if data_dype=='iPhone_static' or data_dype == 'modified_phone_static':
            output_image_static = 0
            total = 0
            for batch, sample in enumerate(loader_test):
                flist = []
                sample = {key: val.cuda() if val is not None and not key == 'name' else val for key, val in sample.items()}
                sample["rgb"] = sample["rgb"].view(-1, 3, sample["rgb"].shape[-2], sample["rgb"].shape[-1])
                sample["dep"] = sample["dep"].view(-1, 1, sample["dep"].shape[-2], sample["dep"].shape[-1])
                sample["rgb"] = sample["rgb"].float()
                output = net(sample)
                output[sample['dep']>0] = sample['dep'][sample['dep']>0]
                pbar.set_description('Testing')
                pbar.update(loader_test.batch_size)
                output_image = output.detach().cpu().numpy()
                if batch%25==12:output_image_static = output_image
                if batch%25==24:
                    for i in range(25):
                        cv2.imwrite(os.path.join(output_path, f'{total+i}.exr'), (output_image_static[0,0,:,:]).astype(np.float32))
                        flist.append(f'{total+i}.exr')  
                    with open(f'{output_path}/data.list', 'a') as f:
                        for item in flist:
                            f.write("%s\n" % item)
                    total+=25
                    output_image_static = 0
            pbar.close()
        else:
            for batch, sample in enumerate(loader_test):
                flist = []
                sample = {key: val.cuda() if val is not None and not key == 'name' else val for key, val in sample.items()}
                sample["rgb"] = sample["rgb"].view(-1, 3, sample["rgb"].shape[-2], sample["rgb"].shape[-1])
                sample["dep"] = sample["dep"].view(-1, 1, sample["dep"].shape[-2], sample["dep"].shape[-1])
                sample["rgb"] = sample["rgb"].float()
                output = net(sample)
                output[sample['dep']>0] = sample['dep'][sample['dep']>0]
                pbar.set_description('Testing')
                pbar.update(loader_test.batch_size)
                output_image = output.detach().cpu().numpy()
                cv2.imwrite(os.path.join(output_path, f'{batch}.exr'), (output_image[0, 0, :, :]).astype(np.float32))
                flist.append(f'{batch}.exr')  
                with open(f'{output_path}/data.list', 'a') as f:
                    for item in flist:
                        f.write("%s\n" % item)
            pbar.close()

def test2(args):
    ### Prepare dataset
    data_types = ['iPhone_dynamic','iPhone_static','modified_phone_static','synthetic']
    for data_dype in data_types:
        data_test = DataLoad(args, 'test',data_dype)
        loader_test = DataLoader(dataset=data_test, batch_size=1, shuffle=False)
        ###  Set up Network
        net = UNet(in_planes=4)
        net.cuda()
        if args.pretrain is not None:
            assert os.path.exists(args.pretrain), \
                "file not found: {}".format(args.pretrain)

            checkpoint = torch.load(args.pretrain)
            net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()},strict=False)
            # net.load_state_dict(checkpoint['net'], strict=True)
        net.eval()

        num_sample = len(loader_test)*loader_test.batch_size

        output_path = os.path.join( './result',data_dype)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        pbar = tqdm(total=num_sample)
        if data_dype=='iPhone_static' or data_dype == 'modified_phone_static':
            output_image_static = 0
            total = 0
            for batch, sample in enumerate(loader_test):
                flist = []
                sample = {key: val.cuda() if val is not None and not key == 'name' else val for key, val in sample.items()}
                sample["rgb"] = sample["rgb"].view(-1, 3, sample["rgb"].shape[-2], sample["rgb"].shape[-1])
                sample["dep"] = sample["dep"].view(-1, 1, sample["dep"].shape[-2], sample["dep"].shape[-1])
                sample["rgb"] = sample["rgb"].float()
                output = net(sample)
                output[sample['dep']>0] = sample['dep'][sample['dep']>0]
                pbar.set_description('Testing')
                pbar.update(loader_test.batch_size)
                output_image = output.detach().cpu().numpy()
                output_image_static += output_image
                if batch%25==24:
                    for i in range(25):
                        cv2.imwrite(os.path.join(output_path, f'{total+i}.exr'), (output_image_static[0,0,:,:]/25).astype(np.float32))
                        flist.append(f'{total+i}.exr')  
                    with open(f'{output_path}/data.list', 'a') as f:
                        for item in flist:
                            f.write("%s\n" % item)
                    total+=25
                    output_image_static = 0
            pbar.close()
        else:
            for batch, sample in enumerate(loader_test):
                flist = []
                sample = {key: val.cuda() if val is not None and not key == 'name' else val for key, val in sample.items()}
                sample["rgb"] = sample["rgb"].view(-1, 3, sample["rgb"].shape[-2], sample["rgb"].shape[-1])
                sample["dep"] = sample["dep"].view(-1, 1, sample["dep"].shape[-2], sample["dep"].shape[-1])
                sample["rgb"] = sample["rgb"].float()
                output = net(sample)
                output[sample['dep']>0] = sample['dep'][sample['dep']>0]
                pbar.set_description('Testing')
                pbar.update(loader_test.batch_size)
                output_image = output.detach().cpu().numpy()
                cv2.imwrite(os.path.join(output_path, f'{batch}.exr'), (output_image[0, 0, :, :]).astype(np.float32))
                flist.append(f'{batch}.exr')  
                with open(f'{output_path}/data.list', 'a') as f:
                    for item in flist:
                        f.write("%s\n" % item)
            pbar.close()

def validate(args):
    ### Prepare dataset
    data_types = ['iPhone_dynamic','iPhone_static','modified_phone_static','synthetic']
    for data_dype in data_types:
        data_test = DataLoad(args, 'test',data_dype,validate=True)
        loader_test = DataLoader(dataset=data_test, batch_size=1, shuffle=False)
        ###  Set up Network
        net = UNet(in_planes=4)
        net.cuda()
        if args.pretrain is not None:
            assert os.path.exists(args.pretrain), \
                "file not found: {}".format(args.pretrain)

            checkpoint = torch.load(args.pretrain)
            net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()},strict=False)
            # net.load_state_dict(checkpoint['net'], strict=True)
        net.eval()

        num_sample = len(loader_test)*loader_test.batch_size

        output_path = os.path.join( './validate',data_dype)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        pbar = tqdm(total=num_sample)
        if data_dype=='iPhone_static' or data_dype == 'modified_phone_static':
            output_image_static = 0
            for batch, sample in enumerate(loader_test):
                flist = []
                sample = {key: val.cuda() if val is not None and not key == 'name' else val for key, val in sample.items()}
                sample["rgb"] = sample["rgb"].view(-1, 3, sample["rgb"].shape[-2], sample["rgb"].shape[-1])
                sample["dep"] = sample["dep"].view(-1, 1, sample["dep"].shape[-2], sample["dep"].shape[-1])
                sample["rgb"] = sample["rgb"].float()
                output = net(sample)
                output[sample['dep']>0] = sample['dep'][sample['dep']>0]
                pbar.set_description('Testing')
                pbar.update(loader_test.batch_size)
                output_image = output.detach().cpu().numpy()
                output_image_static += output_image
            for i in range(num_sample):
                cv2.imwrite(os.path.join(output_path, f'{i}.exr'), (output_image_static[0, 0, :, :]/25).astype(np.float32))
                flist.append(f'{i}.exr')  
            with open(f'{output_path}/data.list', 'a') as f:
                for item in flist:
                    f.write("%s\n" % item)
            pbar.close()
        else:
            for batch, sample in enumerate(loader_test):
                flist = []
                sample = {key: val.cuda() if val is not None and not key == 'name' else val for key, val in sample.items()}
                sample["rgb"] = sample["rgb"].view(-1, 3, sample["rgb"].shape[-2], sample["rgb"].shape[-1])
                sample["dep"] = sample["dep"].view(-1, 1, sample["dep"].shape[-2], sample["dep"].shape[-1])
                sample["rgb"] = sample["rgb"].float()
                output = net(sample)
                output[sample['dep']>0] = sample['dep'][sample['dep']>0]
                pbar.set_description('Testing')
                pbar.update(loader_test.batch_size)
                output_image = output.detach().cpu().numpy()
                cv2.imwrite(os.path.join(output_path, f'{batch}.exr'), (output_image[0, 0, :, :]).astype(np.float32))
                flist.append(f'{batch}.exr')  
                with open(f'{output_path}/data.list', 'a') as f:
                    for item in flist:
                        f.write("%s\n" % item)
            pbar.close()

def validate2(args):
    ### Prepare dataset
    data_types = ['iPhone_dynamic','iPhone_static','modified_phone_static','synthetic']
    for data_dype in data_types:
        data_test = DataLoad(args, 'test',data_dype,validate=True)
        loader_test = DataLoader(dataset=data_test, batch_size=1, shuffle=False)
        ###  Set up Network
        net = UNet(in_planes=4)
        net.cuda()
        if args.pretrain is not None:
            assert os.path.exists(args.pretrain), \
                "file not found: {}".format(args.pretrain)

            checkpoint = torch.load(args.pretrain)
            net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()},strict=False)
            # net.load_state_dict(checkpoint['net'], strict=True)
        net.eval()
        net2 = UNet(in_planes=4)
        net2.cuda()
        if args.pretrain is not None:
            assert os.path.exists(args.pretrain), \
                "file not found: {}".format(args.pretrain)
            name = args.pretrain
            names = name.split('/')
            epoch = int(names[-1][6:6+5])-1
            names[-1] = 'model_000'+str(epoch)+'.pt'
            name = '/'.join(names)
            checkpoint = torch.load(name)
            net2.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()},strict=False)
            # net.load_state_dict(checkpoint['net'], strict=True)
        net.eval()
        net2.eval()
        num_sample = len(loader_test)*loader_test.batch_size

        output_path = os.path.join( './validate',data_dype)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        pbar = tqdm(total=num_sample)
        total = 0
        if data_dype=='iPhone_static' or data_dype == 'modified_phone_static':
            output_image_static = 0
            for batch, sample in enumerate(loader_test):
                flist = []
                sample = {key: val.cuda() if val is not None and not key == 'name' else val for key, val in sample.items()}
                sample["rgb"] = sample["rgb"].view(-1, 3, sample["rgb"].shape[-2], sample["rgb"].shape[-1])
                sample["dep"] = sample["dep"].view(-1, 1, sample["dep"].shape[-2], sample["dep"].shape[-1])
                sample["rgb"] = sample["rgb"].float()
                output = net(sample)
                output[sample['dep']>0] = sample['dep'][sample['dep']>0]
                output2 = net2(sample)
                output2[sample['dep']>0] = sample['dep'][sample['dep']>0]
                output = (output+output2)/2
                pbar.set_description('Testing')
                pbar.update(loader_test.batch_size)
                output_image = output.detach().cpu().numpy()
                if batch%25==12:
                    output_image_static = output_image
                if batch%25==24:
                    for i in range(25):
                        cv2.imwrite(os.path.join(output_path, f'{total+i}.exr'), (output_image_static[0, 0, :, :]).astype(np.float32))
                        flist.append(f'{total+i}.exr')  
                    with open(f'{output_path}/data.list', 'a') as f:
                        for item in flist:
                            f.write("%s\n" % item)
                    total+=25
            pbar.close()
        else:
            for batch, sample in enumerate(loader_test):
                flist = []
                sample = {key: val.cuda() if val is not None and not key == 'name' else val for key, val in sample.items()}
                sample["rgb"] = sample["rgb"].view(-1, 3, sample["rgb"].shape[-2], sample["rgb"].shape[-1])
                sample["dep"] = sample["dep"].view(-1, 1, sample["dep"].shape[-2], sample["dep"].shape[-1])
                sample["rgb"] = sample["rgb"].float()
                output = net(sample)
                output[sample['dep']>0] = sample['dep'][sample['dep']>0]
                output2 = net2(sample)
                output2[sample['dep']>0] = sample['dep'][sample['dep']>0]
                output = (output+output2)/2
                pbar.set_description('Testing')
                pbar.update(loader_test.batch_size)
                output_image = output.detach().cpu().numpy()
                cv2.imwrite(os.path.join(output_path, f'{batch}.exr'), (output_image[0, 0, :, :]).astype(np.float32))
                flist.append(f'{batch}.exr')  
                with open(f'{output_path}/data.list', 'a') as f:
                    for item in flist:
                        f.write("%s\n" % item)
            pbar.close()

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
    max_val = np.max(ref_image)
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


def _RMAE(pred_dep, gt):
    rmae = torch.mean(torch.abs((pred_dep[gt>0]-gt[gt>0])/gt[gt>0]))
    return rmae

 # Relative Mean Absolute Error
def RMAE(pred_dep, gt):
    rmae = np.mean(np.abs((pred_dep[gt>0]-gt[gt>0])/gt[gt>0]))
    return rmae

# Relative Depth shift
def RDS(pred_dep, depsp):
    x_sp, y_sp = np.where(depsp>0)
    d_sp = depsp[x_sp, y_sp]
    rds = np.mean(abs(pred_dep[x_sp, y_sp]-d_sp)/d_sp)
    return rds

# Relative Temporal Standard Deviation
def _RTSD(pred_deps,dim=0):
    rtsd = torch.mean(torch.std(pred_deps, dim=0)/(torch.mean(pred_deps,dim=0)+1e-5))
    return rtsd

# Relative Temporal Standard Deviation
def RTSD(pred_deps):
    rtsd = np.mean(np.std(pred_deps, axis=0)/(np.mean(pred_deps,axis=0)+1e-5))
    return rtsd

def visualize(img,name):
    plt.imshow(img,cmap='jet_r')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(name)
    plt.close()

if __name__ == "__main__":
 
    args = get_parser()
    if args.test_only:
        test(args)
    elif args.validate_only:
        validate(args)
    else:
        train(args)