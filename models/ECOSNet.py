import torch
import torch.nn.functional as F
from torch import nn
import math
import random

from .segmentation import build_segmentation_head
from .lstm_backbone import build_recurrent_backbone
from .transformer import build_CSFM


class ECOSNet(nn.Module):
    def __init__(self, backbone, ref_backbone, transformer, segmentation_head, hidden_dim=256):
        super().__init__()
        self.seg_backbone = backbone
        self.ref_backbone = ref_backbone

        self.bottleneck = nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=1)  # the bottleneck layer

        self.transformer = transformer
        self.seg_head = segmentation_head

    def extract_ref_feats(self, ref_pairs, states):
        frame, obj_mask = ref_pairs['frame'], ref_pairs['obj_mask']
        features, states = self.ref_backbone(frame, obj_mask, states)
        return features, states

    def extract_seg_feats(self, seg_frames, states):
        features, states = self.seg_backbone(seg_frames, states)
        f16 = features[-1]
        f16 = self.bottleneck(f16)
        return f16, features, states

    def forward_transformer(self, query, key, value, prev_states=None):
        enc_mem = self.transformer(query, key, value)
        return None, enc_mem, prev_states

    def segment(self, hs, mem, seq_dict, fpns, ref_mask, prev_states=None):
        bs, _, h, w = fpns[-1].shape  # layer3
        _, bs_no, C = mem.shape

        enc_opt = mem.permute(1, 2, 0).contiguous().view(bs_no, C, h, w)
        opt_feat = enc_opt
        no = int(bs_no / bs)

        r1, r2, r3 = fpns
        r3_size, r2_size, r1_size = r3.shape, r2.shape, r1.shape

        r3 = r3.unsqueeze(1).expand(-1, no, -1, -1, -1).reshape(bs_no, *r3_size[1:])
        r2 = r2.unsqueeze(1).expand(-1, no, -1, -1, -1).reshape(bs_no, *r2_size[1:])
        r1 = r1.unsqueeze(1).expand(-1, no, -1, -1, -1).reshape(bs_no, *r1_size[1:])

        spatial_shape = (bs, ) + ref_mask.shape[1:]

        logits, states = self.seg_head(opt_feat, r3, r2, r1, spatial_shape, prev_states)
        return logits, states

    def forward(self, frames, obj_masks, n_objs, prev_states=None):
        B, T = frames.shape[:2]

        logits_list = []
        probs_list = []

        ref_states = None
        seg_states = None
        match_states = None
        head_states = None

        total_value = None
        total_key = None

        k16 = None
        values = None

        for t in range(T):
            if t == 0:
                ref_pairs = {
                    'frame': frames[:, t],  # [B x 5 x H x W]
                    'obj_mask': obj_masks[:, t, 1:]
                }
                v16, ref_states = self.extract_ref_feats(ref_pairs, ref_states)
                v16 = v16.unsqueeze(3)
                total_value = v16
                values = total_value  # B obj_n C T H W
            elif t == 1:
                pass
            else:
                ref_pairs = {
                    'frame': frames[:, t - 1],  # [B x 5 x H x W]
                    'obj_mask': obj_masks[:, t - 1, 1:]
                }
                v16, ref_states = self.extract_ref_feats(ref_pairs, ref_states)
                total_value = torch.cat([total_value, v16.unsqueeze(3)], dim=3)

                idx_list = list(range(t))
                index = idx_list
                if t > 3:
                    idx_list = list(range(1, t - 1))
                    random.shuffle(idx_list)
                    index = [0] + idx_list[:1] + [t - 1]
                values = total_value[:, :, :, index]
                k16 = total_key[:, :, index]

            seg_frame = frames[:, t]  # [B x 5 x H x W]
            f16, median_layers, seg_states = self.extract_seg_feats(seg_frame, seg_states)

            if total_key is None:
                total_key = f16.unsqueeze(2)
                k16 = total_key

            hs, enc_mem, match_states = self.forward_transformer(f16, k16, values, match_states)

            logits, head_states = self.segment(hs, enc_mem, None, median_layers, obj_masks[:, 0], head_states)
            for batch_id, obj_num in enumerate(n_objs):
                logits[batch_id:batch_id + 1, obj_num:] = -1e+10 if logits.dtype == torch.float32 else -1e+4

            logits_list.append(logits)
            probs_list.append(logits.softmax(dim=1))

            if t > 0:
                total_key = torch.cat([total_key, f16.unsqueeze(2)], dim=2)

        pred_logits = torch.stack(logits_list, dim=1)
        pred_probs = torch.stack(probs_list, dim=1)
        return pred_logits, pred_probs, None


def build_ECOS(cfg):
    backbone = build_recurrent_backbone(cfg, 'seg')
    ref_backbone = build_recurrent_backbone(cfg, 'ref')
    transformer = build_CSFM(cfg)
    segmentation_head = build_segmentation_head(cfg)

    model = ECOSNet(backbone, ref_backbone, transformer, segmentation_head)
    return model
