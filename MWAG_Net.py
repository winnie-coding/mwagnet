import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import math

class DWT_2D(nn.Module):
    def __init__(self, wave='haar'):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # x: (B, C, H, W)
        C = x.shape[1]
        w_ll = self.w_ll.expand(C, -1, -1, -1)
        w_lh = self.w_lh.expand(C, -1, -1, -1)
        w_hl = self.w_hl.expand(C, -1, -1, -1)
        w_hh = self.w_hh.expand(C, -1, -1, -1)

        pad_h = x.shape[2] % 2
        pad_w = x.shape[3] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        ll = F.conv2d(x, w_ll, stride=2, groups=C)
        lh = F.conv2d(x, w_lh, stride=2, groups=C)
        hl = F.conv2d(x, w_hl, stride=2, groups=C)
        hh = F.conv2d(x, w_hh, stride=2, groups=C)

        return ll, lh, hl, hh

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
    def forward(self, x):
        out1 = F.gelu(self.conv1(x))
        out2 = F.gelu(self.conv2(out1))
        out2 = out2 + x 
        return out2

class MSWEM(nn.Module):

    def __init__(self, in_channels, wave='haar', levels=2):
        super(MSWEM, self).__init__()
        assert levels >= 1
        self.levels = levels
        self.dwt = DWT_2D(wave)

        self.high_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels * 3, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.GELU(),
                ResBlock(in_channels)
            ) for _ in range(levels)
        ])

        self.low_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            ResBlock(in_channels)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * (levels + 1), in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        
        self.res_weight = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        residual = x
        low = x
        high_feats = []
        shapes = []

        for i in range(self.levels):
            shapes.append(low.shape) 
            ll, lh, hl, hh = self.dwt(low)

            high_cat = torch.cat([lh, hl, hh], dim=1)  # (B, 3C, H/2, W/2)
            high_feat = self.high_blocks[i](high_cat)  # (B, C, H/2, W/2)
            high_feats.append(high_feat)
            low = ll  

        low_feat = self.low_block(low)  # (B, C, H/(2^L), W/(2^L))

        target_H = x.shape[2]
        target_W = x.shape[3]

        upsampled = []
        up_low = F.interpolate(low_feat, size=(target_H, target_W), mode='bilinear', align_corners=False)
        upsampled.append(up_low)

        for j, hf in enumerate(high_feats):
            up = F.interpolate(hf, size=(target_H, target_W), mode='bilinear', align_corners=False)
            upsampled.append(up)

        fused = torch.cat(upsampled, dim=1)  # (B, C*(L+1), H, W)
        out = self.fusion(fused)  # (B, C, H, W)
        out = out * self.res_weight+ residual
        return out

class WED_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 bias=False, wt_levels=2, wt_type='haar', dropout_rate=0.05):
        super(WED_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=kernel_size // 2, stride=1, groups=in_channels, bias=bias)
        self.base_bn = nn.BatchNorm2d(in_channels)
        self.base_scale = nn.Parameter(torch.ones(1))

        self.wave_extractor = MSWEM(in_channels, wave=wt_type, levels=wt_levels)

        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        if stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):
        x_base = self.base_bn(self.base_conv(x)) * self.base_scale
        x_tag = self.wave_extractor(x)
        x_out = x_base + x_tag
        x_out = self.proj(x_out)  
        if self.do_stride is not None:
            x_out = self.do_stride(x_out)
        return x_out


class GCSC(nn.Module):
    def __init__(self, channels, groups=2):
        super().__init__()
        assert channels % 2 == 0
        self.groups = groups
        ch_half = channels // 2
        self.branch = nn.Sequential(
            nn.Conv2d(ch_half, ch_half, 1, bias=False),
            nn.Conv2d(ch_half, ch_half, 3, padding=1, groups=ch_half, bias=False),            
            nn.Conv2d(ch_half, ch_half, 1, bias=False),
        ) 

    def channel_shuffle(self, x):
        b, c, h, w = x.size()
        x = x.view(b, 2, c // 2, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        y1 = self.branch(x1)
        out = torch.cat([y1, x2], dim=1)
        out = self.channel_shuffle(out)
        return out


class WGAG(nn.Module):
    def __init__(self, channels, groups=2, reduction=8, use_wavelet=True, dwt_layer=None):

        super().__init__()
        assert channels % 2 == 0
        self.sg_x = GCSC(channels, groups=groups)
        self.sg_g = GCSC(channels, groups=groups)

        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.GELU()  

        mid = max(channels // 8, 1)  
        self.spatial_reduce = nn.Conv2d(channels, mid, 1, bias=False)
        self.spatial_expand = nn.Conv2d(mid, channels, 1, bias=False)
        self.spatial_bn = nn.BatchNorm2d(channels)

        self.ca_fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=True)
        self.ca_fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=True)

        self.alpha = nn.Parameter(torch.tensor(0.1))  # spatial weight
        self.beta  = nn.Parameter(torch.tensor(0.1))  # channel weight
        self.gamma = nn.Parameter(torch.tensor(0.1))  # wavelet hf weight

        self.sigmoid = nn.Sigmoid()

        self.use_wavelet = use_wavelet
        self.dwt = dwt_layer

    def _wavelet_hf_mask(self, x):
        with torch.no_grad():
            ll, lh, hl, hh = self.dwt(x)  # sizes: H/2, W/2
            hf = (lh + hl + hh) / 3.0  # (B,C,H/2,W/2)
            hf_map = hf.mean(dim=1, keepdim=True)  # (B,1,H/2,W/2)
            hf_map = F.interpolate(hf_map, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            hf_map = torch.tanh(hf_map)  # keep between -1..1
            return hf_map

    def forward(self, x, g):
        # resize g to match x spatial size
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)

        x1 = self.sg_x(x)
        g1 = self.sg_g(g)

        q = self.relu(self.bn(x1 + g1))  # (B,C,H,W)

        s = self.spatial_reduce(q)
        s = self.relu(s)
        s = self.spatial_expand(s)
        s = self.spatial_bn(s)
        A_spatial = self.sigmoid(s)  # (B,C,H,W)

        ch = F.adaptive_avg_pool2d(q, 1)  # (B,C,1,1)
        ch = F.relu(self.ca_fc1(ch))
        ch = self.sigmoid(self.ca_fc2(ch))  # (B,C,1,1)

        # wavelet HF mask
        if self.use_wavelet and (self.dwt is not None):
            hf_mask = self._wavelet_hf_mask(x)  # (B,1,H,W)
            # broadcast to channels
            hf_mask_c = hf_mask.expand(-1, x.shape[1], -1, -1)
        else:
            hf_mask_c = 0.0

        # fuse attentions (broadcast ch to spatial shape)
        ch_spatial = ch.expand(-1, -1, x.shape[2], x.shape[3])  # (B,C,H,W)
        A = self.alpha * A_spatial + self.beta * ch_spatial + self.gamma * hf_mask_c
        A = self.sigmoid(A)  # final gate in (0,1)

        out = x * A 
        return out

class MWAG_Net(nn.Module):
    def __init__(self, in_channels, out_channels, wave_level=2, model_size='small', wt_type='haar'):
        super(MWAG_Net, self).__init__()
        if model_size == 'small':
            num_channels = [32, 64, 128, 256, 512]
        elif model_size == 'mid':
            num_channels = [64, 128, 256, 512, 1024]
        elif model_size == 'large':
            num_channels = [128, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
        
        self.in_conv = nn.Conv2d(in_channels, num_channels[0], kernel_size=1, padding=0, bias=False)
        
        self.encoder1 = self.conv_block(num_channels[0], num_channels[1], wave_level, wt_type)
        self.encoder2 = self.conv_block(num_channels[1], num_channels[2], wave_level, wt_type)
        self.encoder3 = self.conv_block(num_channels[2], num_channels[3], wave_level, wt_type)
        self.encoder4 = self.conv_block(num_channels[3], num_channels[4], wave_level, wt_type)

        self.middle = self.midconv_block(num_channels[4], num_channels[4])

        self.wgag4 = WGAG(num_channels[4], dwt_layer=DWT_2D(wt_type))
        self.wgag3 = WGAG(num_channels[3], dwt_layer=DWT_2D(wt_type))
        self.wgag2 = WGAG(num_channels[2], dwt_layer=DWT_2D(wt_type))
        self.wgag1 = WGAG(num_channels[1], dwt_layer=DWT_2D(wt_type))

        self.decoder4 = self.dconv_block(num_channels[4]*2, num_channels[3], wave_level, wt_type)
        self.decoder3 = self.dconv_block(num_channels[3]*2, num_channels[2], wave_level, wt_type)
        self.decoder2 = self.dconv_block(num_channels[2]*2, num_channels[1], wave_level, wt_type)
        self.decoder1 = self.dconv_block(num_channels[1]*2, num_channels[0], wave_level, wt_type)

        self.final_conv = nn.Conv2d(num_channels[0], out_channels, kernel_size=1, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def midconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def conv_block(self, in_channels, out_channels, wave_level, wt_type):
        return nn.Sequential(
            WED_Conv(in_channels, out_channels, wt_levels=wave_level, wt_type=wt_type),
            nn.MaxPool2d(2)
        )


    def dconv_block(self, in_channels, out_channels, wave_level, wt_type):
        layers = []
        if in_channels != out_channels and in_channels != out_channels * 2:
            layers.append(nn.Conv2d(in_channels, in_channels // 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(in_channels // 2))
            layers.append(nn.GELU())
            in_ch = in_channels // 2
        else:
            in_ch = in_channels

        layers.extend([
            WED_Conv(in_ch, out_channels, wt_levels=wave_level, wt_type=wt_type)
        ])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.in_conv(x)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        middle = self.middle(enc4)

        dec4 = self.decoder4(torch.cat([middle, enc4], 1))
        dec4 = F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=False)

        gated3 = self.wgag3(enc3, dec4)
        dec3 = self.decoder3(torch.cat([dec4, gated3], 1))
        dec3 = F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=False)

        gated2 = self.wgag2(enc2, dec3)
        dec2 = self.decoder2(torch.cat([dec3, gated2], 1))
        dec2 = F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=False)

        gated1 = self.wgag1(enc1, dec2)
        dec1 = self.decoder1(torch.cat([dec2, gated1], 1))
        dec1 = F.interpolate(dec1, scale_factor=2, mode='bilinear', align_corners=False)

        output = self.final_conv(dec1)
        
        return output