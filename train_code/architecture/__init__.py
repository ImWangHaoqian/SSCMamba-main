import torch
from .edsr import EDSR
from .HDNet import HDNet
from .hinet import HINet
from .hrnet import SGN
from .HSCNN_Plus import HSCNN_Plus
from .MIRNet import MIRNet
from .MPRNet import MPRNet
from .MST import MST
from .MST_Plus_Plus import MST_Plus_Plus
from .Restormer import Restormer
from .GMSR import GMSR
from .SSTHyper import SSTHyper
from .SSCMamba import SSCMamba


def model_generator(method, pretrained_model_path=None):
    if method == 'mirnet':
        model = MIRNet(n_RRG=3, n_MSRB=1, height=3, width=1).cuda()
    elif method == 'mst_plus_plus':
        model = MST_Plus_Plus().cuda()
    elif method == 'mst':
        model = MST(dim=31, stage=2, num_blocks=[4, 7, 5]).cuda()
    # 保留旧方法名兼容：将原来的 mst_ss3d_orthodiagonal_repconv 映射到 SSCMamba 实现
    elif method == 'mst_ss3d_orthodiagonal_repconv' or method == 'sscmamba':
        model = SSCMamba().cuda()
    elif method == 'hinet':
        model = HINet(depth=4).cuda()
    elif method == 'mprnet':
        model = MPRNet(num_cab=4).cuda()
    elif method == 'restormer':
        model = Restormer().cuda()
    elif method == 'edsr':
        model = EDSR().cuda()
    elif method == 'hdnet':
        model = HDNet().cuda()
    elif method == 'hrnet':
        model = SGN().cuda()
    elif method == 'hscnn_plus':
        model = HSCNN_Plus().cuda()
    elif method == 'gmsr':
        model = GMSR().cuda()
    elif method == 'ssthyper':
        model = SSTHyper().cuda()
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        state_dict = {
            k.replace('module.', ''): v
            for k, v in checkpoint['state_dict'].items()
        }
        model.load_state_dict(state_dict, strict=True)
    return model
