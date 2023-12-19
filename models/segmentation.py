import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Refine(nn.Module):
    def __init__(self, inplanes, planes):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, size=s.shape[2:], mode='bilinear')
        m = self.ResMM(m)
        return m


class SegmentationHead(nn.Module):
    def __init__(self, c_in, c_in_1, c_in_2, c_out):
        super(SegmentationHead, self).__init__()
        self.convFM = nn.Conv2d(c_in + 1024, c_out, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(c_out, c_out)
        self.RF2 = Refine(c_in_2, c_out)  # 1/16 -> 1/8
        self.RF1 = Refine(c_in_1, c_out)  # 1/8 -> 1/4

        self.pred2 = nn.Conv2d(c_out, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def soft_aggregation(self, ps, max_obj):
        bs, no, H, W = ps.shape
        em = torch.zeros(bs, max_obj, H, W).to(ps.device)
        em[:, 0, :, :] = torch.prod(1 - ps, dim=1)  # bg prob
        em[:, 1:no + 1, :, :] = ps  # obj prob
        em = torch.clamp(em, 1e-7, 1 - 1e-7)
        logit = torch.log((em / (1 - em)))

        return logit

    def forward(self, opt_feat, r3, r2, r1, spatial_shape, prev_states=None):
        bs, max_obj, h, w = spatial_shape

        opt_feat = torch.cat([opt_feat, r3], dim=1)  # bs_obj C h W

        m3 = self.ResMM(self.convFM(opt_feat))
        m2 = self.RF2(r2, m3)  # out: 1/8, 256
        m1 = self.RF1(r1, m2)  # out: 1/4, 256

        p2 = self.pred2(F.relu(m1))

        p = F.interpolate(p2, size=(h, w), mode='bilinear')

        probs = torch.sigmoid(p)
        probs = probs.view(bs, -1, h, w)
        logits = self.soft_aggregation(probs, max_obj)

        return logits, None


def build_segmentation_head(cfg):
    hidden_dim = cfg.MODEL.HIDDEN_DIM
    return SegmentationHead(c_in=hidden_dim, c_in_1=hidden_dim, c_in_2=hidden_dim * 2, c_out=hidden_dim)
