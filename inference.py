import os

import torch
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
import time
import argparse
import logging

from utils import setup_logging, AverageMeter, VideoWriter
from config import cfg
from models import build_model, load_model
from datasets import multibatch_collate_fn, build_dataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser('Testing Mask Segmentation')
    parser.add_argument('--checkpoint',
                        default='checkpoint_iter40000.pth',
                        type=str,
                        help='checkpoint to test the network')
    parser.add_argument('--results', default='results', type=str, help='result directory')
    parser.add_argument('--gpu', default='8', type=str, help='set gpu id to test the network')

    return parser.parse_args()


def main(args):
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    # Data
    print('==> Preparing dataset')

    testset = build_dataset(cfg, 'val', imset='event_test.txt')
    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4, collate_fn=multibatch_collate_fn)

    # Model
    print("==> creating model")
    model = build_model(args)
    print('==> Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # set eval to freeze batchnorm update
    model.eval()
    model.to(device)
    # set testing parameters
    for p in model.parameters():
        p.requires_grad = False

    # Weights
    if args.checkpoint:
        # Load checkpoint.
        logger.info('Loading state dict from: {0}'.format(args.checkpoint))
        model = load_model(model=model, model_file=args.checkpoint)

    # Test
    print('==> Runing model on dataset, totally {:d} videos'.format(len(testloader)))

    test(testloader, model=model, device=device)

    print('==> Results are saved at: {}'.format(args.results))


def one_hot_mask(mask, cls_num=10):
    if len(mask.size()) == 3:
        mask = mask.unsqueeze(1)
    indices = torch.arange(0, cls_num + 1, device=mask.device).view(1, -1, 1, 1)
    return (mask == indices).float()


def test(testloader, model, device, warmup_clip=1):

    data_time = AverageMeter()
    frame_cnt = 0
    pbar = tqdm(enumerate(testloader), total=len(testloader))
    with torch.no_grad():
        for batch_idx, data in pbar:
            frames, masks, objs, infos = data
            frames = frames[0]
            masks = masks[0]
            n_obj = objs[0]
            info = infos[0]

            ref_states = None
            seg_states = None
            match_states = None
            head_states = None

            original_size = info['original_size']
            T, _, _, _ = frames.shape

            total_value = None
            total_key = None

            pbar.desc = 'Runing video {}, objects {:d}'.format(info['name'], n_obj - 1)
            vid_writer = VideoWriter(args.results, info['name'])

            # process reference pairs
            first_pairs = {
                'frame': frames[0:1].to(device),  # [1 x 5 x H x W]
                'obj_mask': masks[0:1, 1:n_obj].to(device)  # [1 x no x H x W]
            }
            v16, ref_states = model.extract_ref_feats(first_pairs, ref_states)
            v16 = v16.unsqueeze(3)
            total_value = v16

            k16, median_layers, seg_states = model.extract_seg_feats(frames[0:1].to(device), seg_states)
            k16 = k16.unsqueeze(2)
            total_key = k16

            vid_writer.write(f'{0:05d}.png', masks[0:1], original_size)

            for i in tqdm(range(1, T), desc='processing ' + info['name']):
                tic = time.time()
                if i > 1:
                    v16, ref_states = model.extract_ref_feats(previous_pairs, ref_states)
                    total_value = torch.cat([total_value, v16.unsqueeze(3)], dim=3)
                    values = total_value[:, :, :, ::5]
                    k16 = total_key[:, :, ::5]

                else:
                    values = total_value
                    k16 = total_key

                f16, median_layers, seg_states = model.extract_seg_feats(frames[i:i + 1].to(device), seg_states)
                hs, enc_mem, match_states = model.forward_transformer(f16, k16, values, match_states)
                logits, head_states = model.segment(hs, enc_mem, None, median_layers, masks[0:1],
                                                    head_states)  # [1, M, H, W]

                out = torch.softmax(logits, dim=1)
                mask = torch.argmax(out, dim=1)
                mask = one_hot_mask(mask)

                total_key = torch.cat([total_key, f16.unsqueeze(2)], dim=2)

                previous_pairs = {'frame': frames[i:i + 1].to(device), 'obj_mask': mask[:, 1:n_obj]}

                toc = time.time()
                data_time.update(toc - tic, n=1)

                vid_writer.write(f'{i:05d}.png', out, original_size)

            torch.cuda.empty_cache()
            frame_cnt += T - 1
        torch.cuda.empty_cache()
        logger.info("Global FPS:{:.1f}".format(frame_cnt / data_time.sum))
    return


if __name__ == '__main__':
    args = parse_args()

    data_name = cfg.DATA.VAL.DATASET_NAME
    print('==> Test dataset: {}'.format(cfg.DATA.VAL.DATASET_NAME))
    # args.results = os.path.join(args.results, data_name)
    print('==> Save directory: {}'.format(args.results))
    if not os.path.exists(args.results):
        os.makedirs(args.results)

    setup_logging(filename=os.path.join(args.results, 'result.txt'), resume=True)
    logger = logging.getLogger(__name__)

    if os.path.isdir(args.checkpoint):
        ckp_dir = args.checkpoint
        ckp_list = os.listdir(ckp_dir)
        for ckp in ckp_list:
            args.checkpoint = os.path.join(ckp_dir, ckp_list)
            main(args)
    else:
        main(args)
