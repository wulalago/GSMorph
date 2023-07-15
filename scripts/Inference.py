import os
import time
import random
import shutil

import cv2
import torch
import nibabel as nib
import numpy as np

from data_func import load_list
from record_func import Plotter
from image_func import intensity_scale
from help_func import read_yaml, make_dir, tensor2array
from model_func import load_model


def main(path):
    cfg = read_yaml(path)

    device = torch.device('cuda:{}'.format(cfg['GPUId']))

    torch.manual_seed(cfg['Seed'])
    random.seed(cfg['Seed'])

    _, _, test_list = load_list(cfg['TextPath'], fixed_time=cfg['FixedTimePoint'], moving_time=cfg['MovingTimePoint'])

    model = load_model(cfg).to(device)

    model.eval()

    load_path = os.path.join(cfg['CheckpointsPath'], 'Weights', 'model_{}.pth.gz'.format(cfg['LoadChoice']))
    model.load_weight(load_path)

    if os.path.exists(os.path.join(cfg['CheckpointsPath'], 'Logs', 'log.npy')):
        plotter = Plotter(os.path.join(cfg['CheckpointsPath'], 'Logs'))
        plotter.buffer = np.load(os.path.join(cfg['CheckpointsPath'], 'Logs', 'log.npy'), allow_pickle=True)[()]
        plotter.send()

    time_list = []

    for batch_idx, (subj_id) in enumerate(test_list):
        print('testing...', subj_id)

        subj_path = os.path.join(cfg['DataPath'], subj_id)
        slice_ids = os.listdir(subj_path)

        temp_img = cv2.imread(os.path.join(subj_path, str(slice_ids[0]), '{}_IMG.png'.format(cfg['MovingTimePoint'])),
                              cv2.IMREAD_GRAYSCALE)

        w, h = temp_img.shape

        if w != cfg['ImgSize'] or h != cfg['ImgSize']:
            continue

        subj_save_path = os.path.join(cfg['CheckpointsPath'], 'Results', subj_id)
        make_dir(subj_save_path)

        start_time = time.time()
        for slice_id in slice_ids:

            moving_img = cv2.imread(os.path.join(subj_path, str(slice_id), '{}_IMG.png'.format(cfg['MovingTimePoint'])),
                                    cv2.IMREAD_GRAYSCALE)
            fixed_img = cv2.imread(os.path.join(subj_path, str(slice_id), '{}_IMG.png'.format(cfg['FixedTimePoint'])),
                                   cv2.IMREAD_GRAYSCALE)

            moving_seg = cv2.imread(os.path.join(subj_path, str(slice_id), '{}_SEG.png'.format(cfg['MovingTimePoint'])),
                                    cv2.IMREAD_GRAYSCALE)

            moving_img = intensity_scale(moving_img)[np.newaxis, np.newaxis, ...]
            fixed_img = intensity_scale(fixed_img)[np.newaxis, np.newaxis, ...]

            moving_img = torch.FloatTensor(moving_img)
            moving_seg = torch.FloatTensor(moving_seg[np.newaxis, np.newaxis, ...])

            fixed_img = torch.FloatTensor(fixed_img)

            fwt = model(moving_img.to(device), fixed_img.to(device))

            moved_img = fwt['Moved']

            moved_seg = model.stn(moving_seg.to(device), fwt['Flow'], mode='nearest')

            moved_seg = tensor2array(moved_seg, True)
            moved_img = tensor2array(moved_img, True)
            flow = tensor2array(fwt['Flow'], True)

            moved_img = np.uint8(moved_img * 255)

            slice_save_path = os.path.join(subj_save_path, str(slice_id))
            os.makedirs(slice_save_path, exist_ok=True)

            shutil.copy(src=os.path.join(subj_path, str(slice_id), '{}_IMG.png'.format(cfg['MovingTimePoint'])),
                        dst=os.path.join(slice_save_path, 'MovingIMG.png'))
            shutil.copy(src=os.path.join(subj_path, str(slice_id), '{}_IMG.png'.format(cfg['FixedTimePoint'])),
                        dst=os.path.join(slice_save_path, 'FixedIMG.png'))
            shutil.copy(src=os.path.join(subj_path, str(slice_id), '{}_SEG.png'.format(cfg['MovingTimePoint'])),
                        dst=os.path.join(slice_save_path, 'MovingSEG.png'))
            shutil.copy(src=os.path.join(subj_path, str(slice_id), '{}_SEG.png'.format(cfg['FixedTimePoint'])),
                        dst=os.path.join(slice_save_path, 'FixedSEG.png'))

            cv2.imwrite(os.path.join(slice_save_path, 'MovedIMG.png'), moved_img)
            cv2.imwrite(os.path.join(slice_save_path, 'MovedSEG.png'), moved_seg)
            flow = np.transpose(flow, axes=(1, 2, 0))
            z_flow = np.zeros_like(flow[..., 0])[..., np.newaxis]
            flow = np.concatenate([flow, z_flow], axis=-1)
            flow_nii = nib.Nifti1Image(flow[:, :, np.newaxis, :], affine=np.eye(4))
            nib.save(flow_nii, os.path.join(slice_save_path, 'Deformation.nii'))
        time_list.append(time.time() - start_time)

    print(np.mean(time_list))
    print(np.std(time_list))


if __name__ == '__main__':
    cfg_path = '../cfg/VoxelMorph.yaml'
    main(cfg_path)
