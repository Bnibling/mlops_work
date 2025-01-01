
# from mit_semseg.config import cfg
from yacs.config import CfgNode as CN

def gen_cfg(weights_encoder_path, weights_decoder_path, num_class):
    temp_cfg = CN()
    temp_cfg.weights_encoder=weights_encoder_path
    temp_cfg.weights_decoder=weights_decoder_path
    temp_cfg.arch_encoder="hrnetv2"
    temp_cfg.arch_decoder="c1"
    temp_cfg.num_class=num_class
    temp_cfg.fc_dim=720
    temp_cfg.preprocess="resize_and_padding"
    temp_cfg.rotate=False
    temp_cfg.flip=False
    temp_cfg.scaling_perc=0.8
    temp_cfg.border_padding=100
    temp_cfg.imgSizes=(300, 375, 450, 525, 600)
    temp_cfg.imgMaxSize=1000
    temp_cfg.padding_constant=32
    return temp_cfg

