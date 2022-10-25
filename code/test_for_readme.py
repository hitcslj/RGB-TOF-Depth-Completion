import time
import torch
import torch.nn as nn
from model.res_backbone import UNet

pretrain = '/data/RGB_TOF/experiments/7.7.baseline_backbone_shortcut/model_00100.pt'
model = UNet(in_planes=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

checkpoint = torch.load(pretrain)
model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()},strict=False)
 
model.eval()

torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

rgb = torch.rand(1, 3, 192,256).to(device)
dep = torch.rand(1, 1, 192,256).to(device)
input =  {'rgb':rgb,'dep':dep}
with torch.no_grad():
    for i in range(100):
        pred = model(input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_stamp = time.time()
    for i in range(100):
        pred = model(input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print((time.time() - time_stamp) / 100)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

print(get_parameter_number(model))