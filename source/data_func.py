import os

import cv2
import torch
from torch.utils.data import Dataset

from image_func import intensity_scale


def load_list(text_path, fixed_time, moving_time):

    train_list = []
    val_list = []
    test_list = []

    train_subjects = open(os.path.join(text_path, "Train.txt"), 'r').readlines()
    val_subjects = open(os.path.join(text_path, "Validation.txt"), 'r').readlines()
    test_subjects = open(os.path.join(text_path, "Test.txt"), 'r').readlines()

    for subj_list, subj_ids in zip([train_list, val_list], [train_subjects, val_subjects]):
        for subject in subj_ids:
            subject = subject.strip()
            subject_dict = {
                "Moving": {'IMG': os.path.join(subject, f'{moving_time}_IMG.png'),
                           'SEG': os.path.join(subject, f'{moving_time}_SEG.png')},
                "Fixed": {'IMG': os.path.join(subject, f'{fixed_time}_IMG.png'),
                          'SEG': os.path.join(subject, f'{fixed_time}_SEG.png')},
            }
            subj_list.append(subject_dict)

    for subject in test_subjects:
        test_list.append(subject.strip())

    print(f"Train and val on {len(train_list)}, {len(val_list)} pairs, test with {len(test_list)} subjects")

    return train_list, val_list, test_list


class RegistrationDataSet(Dataset):
    def __init__(self, data_list, data_root):
        self.data_list = data_list
        self.data_root = data_root
        torch.manual_seed(3407)

    def __getitem__(self, item):
        data_item = self.data_list[item]

        moving_img = cv2.imread(os.path.join(self.data_root, data_item['Moving']['IMG']), cv2.IMREAD_GRAYSCALE)
        fixed_img = cv2.imread(os.path.join(self.data_root, data_item['Fixed']['IMG']), cv2.IMREAD_GRAYSCALE)

        moving_seg = cv2.imread(os.path.join(self.data_root, data_item['Moving']['SEG']), cv2.IMREAD_GRAYSCALE) / 50.
        fixed_seg = cv2.imread(os.path.join(self.data_root, data_item['Fixed']['SEG']), cv2.IMREAD_GRAYSCALE) / 50.

        moving_img = intensity_scale(moving_img)
        fixed_img = intensity_scale(fixed_img)

        moving_img = torch.FloatTensor(moving_img).unsqueeze(0)
        moving_seg = torch.FloatTensor(moving_seg)

        fixed_img = torch.FloatTensor(fixed_img).unsqueeze(0)
        fixed_seg = torch.FloatTensor(fixed_seg)

        data_dict = {
            'Moving': {'IMG': moving_img, 'SEG': moving_seg},
            'Fixed': {'IMG': fixed_img, 'SEG': fixed_seg}
        }

        return data_dict

    def __len__(self):
        return len(self.data_list)
