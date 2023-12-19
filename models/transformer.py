import torch
import torch.nn as nn
from .pdc import DeformLayer


class ShortDeform(nn.Module):
    def __init__(self, d_model=256, d_qk=512, head=1, dropout=0.1):
        super().__init__()
        self.head = head

        self.offset_conv = nn.Conv2d(d_model * 2, d_model, kernel_size=3, padding=1)
        self.pdc = DeformLayer(d_model, d_model)

    # B C H W    # B C H W    B obj C H W
    def forward(self, query, key, value):
        B, obj, C, H, W = value.shape
        concat = torch.cat([query, key], dim=1)
        offset = self.offset_conv(concat).unsqueeze(1).expand(-1, obj, -1, -1, -1)
        out = self.pdc(value.flatten(0, 1), offset.flatten(0, 1))
        out = out.flatten(2).permute(2, 0, 1)

        query = query.unsqueeze(1).expand(-1, obj, -1, -1, -1).flatten(0, 1).flatten(2)
        out = query.permute(2, 0, 1) + out
        return out


class LongAttention(nn.Module):
    def __init__(self, d_model=256, d_qk=512, head=8, dropout=0.1, norm=None):
        super().__init__()
        self.head = head

        self.q_conv = nn.Conv2d(d_model, d_qk, kernel_size=3, padding=1)
        self.k_conv = nn.Conv2d(d_model, d_qk, kernel_size=3, padding=1)
        self.v_conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)

        self.proj = nn.Linear(d_model, d_model)

        if norm == 'GN':
            self.q_norm = nn.GroupNorm(self.head, d_qk)
            self.k_norm = nn.GroupNorm(self.head, d_qk)
            self.v_norm = nn.GroupNorm(self.head, d_model)
        else:
            self.q_norm = nn.BatchNorm2d(d_qk, track_running_stats=False)
            self.k_norm = nn.BatchNorm2d(d_qk, track_running_stats=False)
            self.v_norm = nn.BatchNorm2d(d_model, track_running_stats=False)


    # B C H W      B C T H W      B obj C T H W
    def forward(self, query, key, value):
        B, obj_num, C, T, H, W = value.shape

        query = self.q_conv(query)
        key = self.k_conv(key.permute(0, 2, 1, 3, 4).flatten(0, 1))  # BT C H W
        value = self.v_conv(value.permute(0, 1, 3, 2, 4, 5).flatten(0, 2))  # BobjT C H W

        for_q = self.q_norm(query)
        for_k = self.k_norm(key).view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)
        for_v = self.v_norm(value).view(B, obj_num, T, -1, H, W).permute(0, 1, 3, 2, 4, 5)

        for_q = for_q.flatten(2).permute(2, 0, 1)  # hw b c
        for_k = for_k.flatten(2).permute(2, 0, 1)  # THW b c
        for_v = for_v.flatten(0, 1).flatten(2).permute(2, 0, 1)  # THW B_obj c

        HW, BN, C = for_q.shape
        THW, _, _ = for_k.shape

        # BN 8 HW C/8
        q = for_q.view(HW, BN, self.head, -1).permute(1, 2, 0, 3).contiguous()

        # BN 8 C/8 THW
        k = for_k.view(THW, BN, self.head, -1).permute(1, 2, 3, 0).contiguous()
        v = for_v.view(THW, BN, obj_num, -1).view(THW, BN, obj_num, self.head, -1)
        v = v.permute(1, 3, 0, 2, 4).flatten(3).contiguous()

        atten = q @ k / (q.shape[-1]**0.5)

        atten = torch.softmax(atten, dim=-1)

        value = atten @ v  # BN head HW obj_c/head
        value = value.view(BN, self.head, HW, obj_num, -1).permute(2, 0, 3, 1, 4).flatten(3)
        value = value.flatten(1, 2)
        out1 = self.proj(value)
        return out1


class CSFM(nn.Module):
    def __init__(self, norm=None):
        super().__init__()
        self.long = LongAttention(norm=norm)
        self.fuse = ShortDeform()

    def forward(self, query, key, value):
        long_value = self.long(query, key, value)
        deform_value = self.fuse(query, key[:, :, -1], value[:, :, :, -1])

        out = long_value + deform_value
        return out


def build_CSFM(cfg):
    return CSFM(cfg.MODEL.NORM)