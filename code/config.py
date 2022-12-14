"""
    All of the parameters are defined here.
"""

import argparse
def get_parser():
    parser = argparse.ArgumentParser(description='RGBD fusion')
    # Dataset
    parser.add_argument('--data_dir',
                        type=str,
                        default='/data/RGB_TOF/train', 
                        help='path to dataset')
    parser.add_argument('--data_list',
                        type=str,
                        default='data.list',
                        help='data list')
    # Hardware
    parser.add_argument('--seed',
                        type=int,
                        default=7240,
                        help='random seed point')
    parser.add_argument('--num_gpus',
                        type=int,
                        default=4,
                        help='number of visible GPUs')
    parser.add_argument('--port',
                        type=str,
                        default='3009',
                        help='multiprocessing port')
    parser.add_argument('--num_threads',
                        type=int,
                        default=4,
                        help='number of threads')
    parser.add_argument('--local_rank', 
                        type=int,
                        default=0)
    # Training
    parser.add_argument('--loss',
                        type=str,
                        default='1.0*L1+1.0*L2',
                        help='loss function configuration')
    parser.add_argument('--pretrain',
                        type=str,
                        default=None,
                        help='path of pretrained model')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='number of epochs to train')
    parser.add_argument('--batch_size',
                        type=int,
                        default=60,
                        help='input batch size for training')
    parser.add_argument('--im_num',
                        type=int,
                        default=5)
    parser.add_argument('--max_depth',
                        type=float,
                        default=10.0,
                        help='maximum depth')
    parser.add_argument('--depth_norm',
                        action='store_true',
                        default=True,
                        help='Normalize spot depth')
    parser.add_argument('--augment',
                        type=bool,
                        default=True,
                        help='data augmentation')
    parser.add_argument('--num_spot',
                        type=int,
                        default=1000,
                        help='number of sparse depth spots')
    parser.add_argument('--grid_spot',
                        action='store_true',
                        default=True,
                        help='random spots by grid')
    parser.add_argument('--cut_mask',
                        action='store_true',
                        default=False,
                        help='randomly cut parts of the input spots')
    parser.add_argument('--noise',
                        type=float,
                        default=0.1,
                        help='stddev of guassian noise augmentation')
    parser.add_argument('--rgb_noise',
                        type=float,
                        default=0.1,
                        help='stddev of rgb guassian noise augmentation')
    # args = parser.parse_args()
    # Testing
    parser.add_argument('--test_only',
                        action='store_true',
                        default=False,
                        help='test only flag')
    parser.add_argument('--validate_only',
                        action='store_true',
                        default=False,
                        help='test only flag')
    parser.add_argument('--test_name',
                        type=str,
                        default='synthetic',
                        help='file name for saving testing results')
    # Optimizer
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='learning rate')
    parser.add_argument('--decay',
                        type=str,
                        default='10,20,30,40,50',
                        help='learning rate decay schedule')
    parser.add_argument('--freeze_bn',
                        action='store_true',
                        default=False,
                        help='freeze bn')
    parser.add_argument('--gamma',
                        type=str,
                        default='1.0,0.5,0.25,0.125,0.0625',
                        help='learning rate multiplicative factors')
    parser.add_argument('--optimizer',
                        default='ADAMW',
                        choices=('SGD', 'ADAM', 'RMSprop','ADAMW'),
                        help='optimizer to use (SGD | ADAM | RMSprop)')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='SGD momentum')
    parser.add_argument('--betas',
                        type=tuple,
                        default=(0.9, 0.999),
                        help='ADAM beta')
    parser.add_argument('--epsilon',
                        type=float,
                        default=1e-8,
                        help='ADAM epsilon for numerical stability')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0,
                        help='weight decay')
    parser.add_argument('--warm_up',
                        action='store_true',
                        default=False,
                        help='do lr warm up during the 1st epoch')
    parser.add_argument('--warm_up_epochs',
                        type=int,
                        default=10,
                        help='warm_up_epochs')
    # Logs
    parser.add_argument('--save_dir',
                        type=str,
                        default='/data/RGB_TOF/experiments_finalTest/',
                        help='dir root  name to save')
    parser.add_argument('--save',
                        type=str,
                        default='baseline',
                        help='dir name to save')
    parser.add_argument('--save_full',
                        action='store_true',
                        default=False,
                        help='save optimizer, scheduler and amp in '
                            'checkpoints (large memory)')
    parser.add_argument('--save_image',
                        action='store_true',
                        default=False,
                        help='save images for test')
    parser.add_argument('--save_result_only',
                        action='store_true',
                        default=False,
                        help='save result images only with submission format')
    args = parser.parse_args()
    
    args.save_dir += args.save
    # current_time = time.strftime('%y%m%d_%H%M%S_')
    return args