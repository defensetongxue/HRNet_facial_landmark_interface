# predict for custom dataset in facial landmark
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

from Dataset import CUSTOM
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import models as models
from config import config, update_config
from utils import decode_preds,visualize_and_save_landmarks
def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        default="./setting/config.yaml", type=str)
    parser.add_argument('--model-file', help='model parameters',
                         default="./setting/HR18-WFLW.pth", type=str)
    parser.add_argument('--data_root', help='path to data',default="./data", type=str)
    parser.add_argument('--save_dir', help='path to data',default="./save_dir", type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args



args = parse_args()

config.freeze()
model = models.get_face_alignment_net(config)
gpus = list((0,)) # just using one gpu for interfence
model = nn.DataParallel(model, device_ids=gpus).cuda()
# load model
state_dict = torch.load(args.model_file)
if 'state_dict' in state_dict.keys():
    state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
else:
    model.module.load_state_dict(state_dict)

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

test_loader = DataLoader(
    dataset=CUSTOM(args.data_root),
    batch_size=1,
    shuffle=False,
    num_workers=1
)
with torch.no_grad():
    for i, (inp, _,image_name) in enumerate(test_loader):
        output = model(inp)
        score_map = output.data.cpu()
        preds = decode_preds(score_map , [56, 56])
        visualize_and_save_landmarks(
            image_path=os.path.join(args.data_root,(image_name[0])),
            preds=preds,
            save_path=os.path.join(args.save_dir,(image_name[0])))




