import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import os
import glob
import logging
import warnings

warnings.filterwarnings('ignore')

import shutil
import argparse
from tqdm import tqdm
import time

from datasets import build_dataset, multibatch_collate_fn
from utils import Loss, AverageMeter, LR_Manage, setup_logging, setup_seed
from config import cfg

from models import build_model, load_model, load_checkpoint
from spikingjelly.clock_driven import functional


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', default='', help='path to the pretrained checkpoint')
    parser.add_argument('--seed', default=1024, type=int)

    parser.add_argument('--exp_name', default='ecos', help='exp name')
    parser.add_argument('--log_dir', default='logs/', help='log_dir file')
    parser.add_argument('--gpu', type=str, default='8', help='gpu id')

    return parser.parse_args()


def main():
    log_dir = args.log_dir

    summary_writer = SummaryWriter(log_dir=log_dir)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if cuda:
        torch.backends.cudnn.benchmark = True
        current_device = torch.cuda.current_device()
        logger.info("Running on " + torch.cuda.get_device_name(current_device))
    else:
        logger.info("Running on CPU")

    model = build_model(args)
    model.to(device)
    logger.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6))

    train_dataset = build_dataset(cfg, mode='train', imset='event_train.txt')

    train_dataloader = data.DataLoader(dataset=train_dataset,
                                       batch_size=cfg.TRAIN.BATCH_SIZE,
                                       num_workers=cfg.TRAIN.NUM_WORKER,
                                       shuffle=True,
                                       drop_last=True,
                                       pin_memory=True,
                                       collate_fn=multibatch_collate_fn)

    dataloaders = {'train': train_dataloader}

    params = [params for name, params in model.named_parameters() if params.requires_grad and 'seg_backbone' not in name]
    seg_params = [params for name, params in model.named_parameters() if params.requires_grad and 'seg_backbone' in name]
    optimizer = torch.optim.AdamW([{
        'params': params,
        'lr': cfg.TRAIN.LR,
        'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
        'names': "params"
    }, {
        'params': seg_params,
        'lr': cfg.TRAIN.LR,
        'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
        'names': "other_params"
    }])

    start_epoch = 0
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            logger.info('Loading state dict from: {0}'.format(args.checkpoint))
            model, optimizer, start_epoch = load_checkpoint(model=model, model_file=args.checkpoint, optimizer=optimizer)
        else:
            raise ValueError("Cannot find model file at {}".format(args.checkpoint))

    train(args, model, train_dataloader, optimizer, device, summary_writer, start_epoch)


def train(args, model, train_dataloader, optimizer, device, summary_writer, start_epoch=0):
    cls_meter = {'train': AverageMeter()}
    iou_meter = {'train': AverageMeter()}
    loss_meter = {'train': AverageMeter()}

    dataloaders = {'train': train_dataloader}

    tblog_iter = {'train': 200}
    log_iter = {'train': len(train_dataloader) * start_epoch}

    all_iter = {'train': cfg.TRAIN.ITER}
    max_itr = all_iter['train']

    cfg.TRAIN.EPOCH = cfg.TRAIN.ITER // len(train_dataloader) + 1

    lr_manage = LR_Manage(base_lr=cfg.TRAIN.LR, max_itr=max_itr, min_lr=cfg.TRAIN.LR / 10)
    lossfn = Loss()
    lr = 0
    logger.info(max_itr)
    phases = ['train']
    for epoch in range(start_epoch, cfg.TRAIN.EPOCH):
        for phase in phases:
            if phase == 'train':
                model.train()

            logger.info("Phase: {} Epoch: {} | {}".format(phase, epoch + 1, cfg.TRAIN.EPOCH))
            pbar = tqdm(dataloaders[phase])
            for (frames, masks, objs, infos) in pbar:
                if log_iter[phase] % tblog_iter[phase] == 0:
                    cls_meter[phase].reset()
                    iou_meter[phase].reset()
                    loss_meter[phase].reset()

                log_iter[phase] += 1

                # N B C H W
                frames = frames.to(device)
                masks = masks.to(device)

                states = None
                preds = list()
                optimizer.zero_grad()

                pred_logits, pred_probs, states = model(frames, masks, objs, states)

                loss, loss_stats = lossfn(pred_logits, pred_probs, masks, objs, log_iter[phase])
                if phase == 'train':
                    lr = lr_manage.adjust_learning_rate(optimizer, log_iter[phase])
                    torch.autograd.backward(loss)
                    optimizer.step()

                cls_meter[phase].update(loss_stats['cls_loss'])
                iou_meter[phase].update(loss_stats['iou_loss'])
                loss_meter[phase].update(loss.item())

                desc = f'[LR:{lr:.7f} | {phase}_Loss:{loss_meter[phase].avg:.4f} | {phase}_CLS:{cls_meter[phase].avg:.4f} | {phase}_IOU:{iou_meter[phase].avg:.4f}]'
                pbar.desc = desc
                if log_iter[phase] % tblog_iter[phase] == 0:
                    summary_writer.add_scalar(f'{phase}_Loss', loss_meter[phase].avg, log_iter[phase] // tblog_iter[phase])
                    summary_writer.add_scalar(f'{phase}_CLS', cls_meter[phase].avg, log_iter[phase] // tblog_iter[phase])
                    summary_writer.add_scalar(f'{phase}_IOU', iou_meter[phase].avg, log_iter[phase] // tblog_iter[phase])
                    logger.info(
                        f'Phase: {phase} Epoch: [{epoch + 1} | {cfg.TRAIN.EPOCH}] | Iter: [{log_iter[phase]} | {all_iter[phase]}] | '
                        + desc)

                if (log_iter[phase]) % cfg.TRAIN.SKIP_SAVE == 0:
                    # save model
                    model_path = os.path.join(args.checkpoint_dir, "checkpoint_iter{}.pth".format(log_iter[phase]))
                    torch.save(
                        {
                            'epoch': epoch + 1,
                            'iter': log_iter[phase],
                            'state_dict': model.state_dict(),
                            'train_loss': loss_meter['train'].avg,
                            'train_CLS': cls_meter['train'].avg,
                            'train_IOU': iou_meter['train'].avg,
                            'optimizer': optimizer.state_dict()
                        }, model_path)
                    logger.info("Backup model at: {}".format(model_path))


if __name__ == "__main__":
    args = parse_args()

    # Set seed
    setup_seed(args.seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Logs
    prefix = args.exp_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '-%Y%m%d-%H%M%S')))
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path)

    scripts_to_save = ['train.py', 'config.py', 'inference.py']
    scripts_to_save += list(glob.glob(os.path.join('models', '*.py')))
    scripts_to_save += list(glob.glob(os.path.join('datasets', '*.py')))
    scripts_to_save += list(glob.glob(os.path.join('utils', '*.py')))

    for script in scripts_to_save:
        dst_path = os.path.join(script_path, script)
        try:
            shutil.copy(script, dst_path)
        except IOError:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(script, dst_path)

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    args.checkpoint_dir = checkpoint_dir

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))

    logger = logging.getLogger(__name__)
    logger.info('Config: {}'.format(cfg))
    logger.info('Arguments: {}'.format(args))
    logger.info('Experiment: {}'.format(args.exp_name))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main()
