import sys

sys.path.append('../source')

import os
import shutil
import random

import wandb
import torch
from torch.utils.data import DataLoader

from eval_func import seg_eval, sim_eval
from model_func import load_model, GradientDescent
from help_func import read_yaml, make_dir, tensor2array
from record_func import Recorder
from data_func import load_list, RegistrationDataSet


# This only be used when debug the code
# The wandb won't upload the log into the server
# os.environ['WANDB_MODE'] = 'offline'


def train(updater, train_loader, epoch, sess):
    running_loss = 0.
    print_freq = max(len(train_loader.dataset) // train_loader.batch_size // 4, 1)
    for batch_idx, batch in enumerate(train_loader):
        fwd = updater.model(batch['Moving']['IMG'].to(updater.device), batch['Fixed']['IMG'].to(updater.device))

        loss_total, loss_info, print_info = updater.update_gradient(fwd)

        running_loss += loss_total
        sess.log(loss_info, step=epoch)

        if batch_idx % print_freq == 0 and batch_idx != 0:
            print('[Epoch {:3d} : {:3d}%] {}'.format(
                epoch, round(batch_idx * 100 / len(train_loader)), print_info
            ))

    running_loss /= len(train_loader)
    return running_loss


def val(model, val_loader, epoch, sess):
    running_seg = 0.
    running_sim = 0.
    recorder = Recorder()

    for batch_idx, batch in enumerate(val_loader):
        eval_dict = {}

        fwd = model(batch['Moving']['IMG'].to(model.model_device), batch['Fixed']['IMG'].to(model.model_device))

        moved_img = fwd['Moved']

        moved_seg = model.stn(batch['Moving']['SEG'].unsqueeze(0).to(model.model_device), fwd['Flow'], mode='nearest')

        moved_seg = tensor2array(moved_seg, True)
        moved_img = tensor2array(moved_img, True)

        fixed_img = tensor2array(batch['Fixed']['IMG'], True)
        fixed_seg = tensor2array(batch['Fixed']['SEG'], True)

        seg_dict = seg_eval(moved_seg, fixed_seg)
        sim_dict = sim_eval(moved_img, fixed_img)

        eval_dict.update(seg_dict)
        eval_dict.update(sim_dict)

        running_seg += seg_dict['AvgDice']
        running_sim += sim_dict['MSE']

        recorder.update(eval_dict)

    running_seg /= len(val_loader)
    running_sim /= len(val_loader)

    sess.log(recorder.average(), step=epoch)
    return recorder.info(), running_seg


def main(path):
    cfg = read_yaml(path)

    sess = wandb.init(project="GradientSurgery", name=os.path.basename(cfg['CheckpointsPath']))
    device = torch.device('cuda:{}'.format(cfg['GPUId']))

    torch.manual_seed(cfg['Seed'])
    random.seed(cfg['Seed'])

    train_list, val_list, _ = load_list(cfg['TextPath'],
                                        fixed_time=cfg['FixedTimePoint'], moving_time=cfg['MovingTimePoint'])

    make_dir(cfg['CheckpointsPath'])
    make_dir(os.path.join(cfg['CheckpointsPath'], 'Logs'))
    make_dir(os.path.join(cfg['CheckpointsPath'], 'Weights'))
    make_dir(os.path.join(cfg['CheckpointsPath'], 'Results'))

    shutil.copy(path, os.path.join(cfg['CheckpointsPath'], 'Logs', 'Configuration.yaml'))

    model = load_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['LearningRate'], weight_decay=cfg['WeightDecay'])
    gradient_descent = GradientDescent(model, optimizer,
                                       sim_loss=cfg['SimLoss'], reg_loss=cfg['RegLoss'], loss_weight=cfg['LossWeight'],
                                       gradient_surgery=cfg['GradientSurgery'])

    if cfg['InitWeight']:
        model.load_weight(cfg['InitWeight'])
        print('>>> Load from', cfg['InitWeight'])

    train_set = RegistrationDataSet(train_list, cfg['DataPath'])
    val_set = RegistrationDataSet(val_list, cfg['DataPath'])

    train_loader = DataLoader(train_set, batch_size=cfg['BatchSize'], num_workers=cfg['NumWorkers'],
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    best_metric = 0.
    best_epoch = 0

    for epoch in range(cfg['StartEpoch'], cfg['NumEpoch']+cfg['StartEpoch']):
        model.train()

        running_loss = train(gradient_descent, train_loader, epoch, sess)

        if epoch % cfg['ValFreq'] == 0:
            model.eval()
            running_info, running_metric = val(model, val_loader, epoch, sess)

            print('Epoch {} >>> Loss: {:.4f} Metric: {} Best Epoch: {:03d} with Best Metric: {:.5f}'.format(
                epoch, running_loss, running_info, best_epoch, best_metric))
            open(os.path.join(cfg['CheckpointsPath'], 'Logs', 'record.txt'), 'a+').write(f'{epoch}-{running_loss}-{running_info}\n')

            if running_metric >= best_metric:
                torch.save(model.state_dict(), os.path.join(cfg['CheckpointsPath'], 'Weights', 'model_best.pth.gz'))
                best_metric = running_metric
                best_epoch = epoch

            sess.log({'BestEpoch': best_epoch, 'BestMetric': best_metric}, step=epoch)

        if cfg['EfficientSave']:
            torch.save(model.state_dict(), os.path.join(cfg['CheckpointsPath'], 'Weights', 'model_last.pth.gz'))
        else:
            torch.save(model.state_dict(), os.path.join(cfg['CheckpointsPath'], 'Weights', f'model_{epoch}.pth.gz'))


if __name__ == '__main__':
    cfg_path = '../cfg/VoxelMorph.yaml'
    main(cfg_path)
