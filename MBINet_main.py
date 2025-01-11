import torch
from torch import nn
from contextpath import context_path
from ghostpath import ghost_path
import warnings
import time

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class Channel_Att(nn.Module):
    def __init__(self, channels):
        super(Channel_Att, self).__init__()
        self.channels = channels

        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * residual  #

        return x


class NAMAttention(nn.Module):
    def __init__(self, channels):
        super(NAMAttention, self).__init__()
        self.Channel_Att = Channel_Att(channels)

    def forward(self, x):
        x_out1 = self.Channel_Att(x)

        return x_out1

class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_channels += 2
        if with_r:
            in_channels += 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding = padding)

    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x

class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.ca1 = NAMAttention(channels=128)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.ca2 = NAMAttention(channels=256)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)
        self.CoordConv1 = CoordConv(64, 128, 3, 2, 1)
        self.CoordConv2 = CoordConv(128, 256, 3, 2, 1)

    def forward(self, input):
        x = self.convblock1(input) #(2, 64, 128, 128)
        x = self.CoordConv1(x)
        x = self.ca1(x)# 缺注意力
        x = self.CoordConv2(x)
        x = self.ca2(x)# 缺注意力
        return x

def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))

class Attention(nn.Module):
    def __init__(self, in_places, scale=2, eps=1e-6):
        super(Attention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (self.gamma * weight_value).contiguous()

class ARM(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_atten = Attention(out_channels)


    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        att = self.bn(self.conv_atten(x))
        # channels of input and x should be same
        x = torch.mul(input, att)
        return x


class MFAM(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_atten = Attention(num_classes)
        self.bn = nn.BatchNorm2d(num_classes)

    def forward(self, input_1, input_2, input_3):
        x = torch.cat((input_1, input_2, input_3), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)  #论文为全局池化

        x = self.relu(self.conv1(x))
        x = self.bn(self.conv_atten(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


class BiSeNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path()

        # build context path
        self.context_path = context_path()

        self.ghost_path = ghost_path()
        # build attention refinement module  for resnet 101

        # build attention refinement module  for resnet 18
        self.attention_refinement_module1 = ARM(64, 64)
        self.attention_refinement_module2 = ARM(128, 128)
        self.attention_refinement_module3 = ARM(256, 256)
        self.attention_refinement_module4 = ARM(512, 512)
        # supervision block
        self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
        self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        # build feature fusion module
        self.feature_fusion_module = MFAM(num_classes, 1472)

        # build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

        self.init_weight()

        self.mul_lr = []
        self.mul_lr.append(self.saptial_path)
        self.mul_lr.append(self.attention_refinement_module1)
        self.mul_lr.append(self.attention_refinement_module2)
        self.mul_lr.append(self.supervision1)
        self.mul_lr.append(self.supervision2)
        self.mul_lr.append(self.feature_fusion_module)
        self.mul_lr.append(self.conv)

    def init_weight(self):
        for name, m in self.named_modules():
            if 'context_path' not in name:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-5
                    m.momentum = 0.1
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        # output of spatial path
        sx = self.saptial_path(input)

        gx = self.ghost_path(input)

        # output of context path
        cx1, cx2, cx3, cx4, tail = self.context_path(input)
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx3 = self.attention_refinement_module3(cx3)
        cx4 = self.attention_refinement_module4(cx4)
        cx4 = torch.mul(cx4, tail)
        # upsampling
        cx1 = torch.nn.functional.interpolate(cx1, size=gx.size()[-2:], mode='bilinear')
        cx2 = torch.nn.functional.interpolate(cx2, size=gx.size()[-2:], mode='bilinear')
        cx3 = torch.nn.functional.interpolate(cx3, size=gx.size()[-2:], mode='bilinear')
        cx4 = torch.nn.functional.interpolate(cx4, size=gx.size()[-2:], mode='bilinear')
        cx = torch.cat((cx1, cx2, cx3, cx4), dim=1)

        result = self.feature_fusion_module(sx, cx, gx)

        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')

        result = self.conv(result)


        if self.training:
            return result  # , cx1_sup, cx2_sup

        return result


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())

    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    return {'Total': total_num / 1000000.0, 'Trainable': trainable_num / 1000000.0}
