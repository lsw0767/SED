import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from torch.autograd import Variable

LOWER_BOUND = 1e-2
SQUEEZED_BIN = 8


def move_data_to_gpu(x, cuda):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class GWRP(nn.Module):
    def __init__(self, weight_len):
        super(GWRP, self).__init__()
        decay = LOWER_BOUND**(1/(weight_len-1.))
        self.gwrp_w = decay ** np.arange(weight_len)

        self.gwrp_w = torch.Tensor(self.gwrp_w).cuda()
        self.sum_gwrp_w = torch.sum(self.gwrp_w)

    def forward(self, input):
        x = input.view((input.shape[0], input.shape[1],
                        input.shape[2] * input.shape[3]))

        (x, _) = torch.sort(x, dim=-1, descending=True)
        x = x * self.gwrp_w[None, None, :]
        x = torch.sum(x, dim=-1)

        output = x / self.sum_gwrp_w
        output = output.view(output.shape[0], output.shape[1])
        return output


class RGWRP(nn.Module):
    def __init__(self, weight_len, R_factor=0.221):
        #   mean+1sigma r factor
        #   0.221 for original, random_clip
        #   0.624 for no_clipping
        #   0.597 for mix
        super(RGWRP, self).__init__()
        weight_len = int(weight_len*R_factor)
        # weight_len = 750
        decay = LOWER_BOUND**(1/(weight_len-1.))
        self.gwrp_w = decay ** np.arange(weight_len)

        self.gwrp_w = torch.Tensor(self.gwrp_w).cuda()
        self.sum_gwrp_w = torch.sum(self.gwrp_w)
        self.sort_len = weight_len

    def forward(self, input):
        x = input.view((input.shape[0], input.shape[1],
                        input.shape[2] * input.shape[3]))

        (x, _) = torch.topk(x, k=self.sort_len)

        x = x * self.gwrp_w[None, None, :]
        x = torch.sum(x, dim=-1)

        output = x / self.sum_gwrp_w
        output = output.view(output.shape[0], output.shape[1])
        return output


class TP(nn.Module):
    def __init__(self):
        super(TP, self).__init__()
        # self.alpha = nn.Parameter(torch.tensor(0., dtype=torch.float32).cuda())
        # self.alpha = 0.2
        self.alpha = -0.4


    def forward(self, input):
        output = input.view((input.shape[0], input.shape[1],
                             input.shape[2] * input.shape[3]))

        threshold = output.mean(2, keepdim=True)+self.alpha*output.std(2, keepdim=True)
        # threshold = output.mean(2, keepdim=True)
        output = torch.relu(output-threshold)
        non_zero = torch.tensor((output>0).sum(2)+1, dtype=torch.float32).cuda()
        output = output.sum(2)/non_zero+threshold.squeeze(2)
        output = torch.clamp(output, 0., 1.)

        output = output.view(output.shape[0], output.shape[1])
        return output


class GT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        output = input.view((input.shape[0], input.shape[1],
                             input.shape[2] * input.shape[3]))

        # threshold = output.mean(2, keepdim=True)+alpha*output.std(2, keepdim=True)
        threshold = output.mean(2, keepdim=True)
        t5_threshold, t5_idx = torch.topk(threshold, 3, dim=1)
        mask = torch.zeros_like(threshold)
        mask[:, t5_idx, :] = 1
        threshold = threshold*mask
        output = torch.relu(output-threshold)
        non_zero = torch.tensor((output!=0).sum(2)+1, dtype=torch.float32).cuda()

        ctx.save_for_backward(input, non_zero)

        output = output.sum(2)/non_zero+threshold.squeeze(2)
        output = torch.clamp(output, 0., 1.)
        ctx.save_for_backward(output, non_zero)

        output = output.view(output.shape[0], output.shape[1])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, none_zero = ctx.saved_variables
        grad_input = grad_output.clone()
        grad_input = (grad_input/none_zero).unsqueeze(-1).unsqueeze(-1)
        grad_input = grad_input.unsqueeze(0).repeat(1, 1, 1, 311, 8).view(-1, 41, 311, 8)

        _, idx = torch.topk(output, 3, dim=1)
        grad_alpha = grad_output.clone()
        grad_alpha = torch.gather(grad_alpha, 1, idx)
        return grad_input, grad_alpha


class AlphaMEX(nn.Module):
    def __init__(self):
        super(AlphaMEX, self).__init__()
        self.alpha = nn.Parameter(torch.stack([torch.tensor(0.8, dtype=torch.float32).cuda() for _ in range(41)], dim=0))

    def forward(self, input):
        input = input.view((input.shape[0], input.shape[1],
                             input.shape[2] * input.shape[3]))

        alpha = torch.clamp(self.alpha, 0., 0.9999)
        alpha = (alpha/(1-alpha)).unsqueeze(0).unsqueeze(2)
        mean = torch.pow(alpha.repeat([input.shape[0], 1, input.shape[2]]), input).mean(2)
        log = torch.log(mean)
        am = 1/torch.log(alpha.squeeze(2))*log

        return am


class MEX(nn.Module):
    def __init__(self):
        super(MEX, self).__init__()
        self.alpha = nn.Parameter(torch.stack([torch.tensor(1., dtype=torch.float32).cuda() for _ in range(41)], dim=0))

    def forward(self, input):
        input = input.view((input.shape[0], input.shape[1],
                             input.shape[2] * input.shape[3]))

        alpha = self.alpha.unsqueeze(0).unsqueeze(2)
        exp = torch.exp(self.alpha.unsqueeze(0).unsqueeze(2)*input)
        mean = exp.mean(2)
        log = torch.log(mean)
        am = 1/alpha.squeeze(2)*log

        return am

class VggishConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VggishConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=True)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input):
        x = input
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x


class VggishDeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(2, 2)):
        super(VggishDeConvBlock, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(3, 3), stride=stride, padding=(1, 1),
                                        output_padding=(1, 1) if stride==(2, 2) else (1, 0),
                                        bias=True)

        self.conv2 = nn.ConvTranspose2d(in_channels=out_channels,
                                        out_channels=out_channels,
                                        kernel_size=(3, 3), stride=(1, 1),
                                        padding=(1, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input):
        x = input
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x


class MixupBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MixupBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, 64), stride=(1, 64),
                               # output_padding=(1, 32),
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, 64), stride=(1, 64),
                               # output_padding=(1, 32),
                               bias=True)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, input):
        x = input
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.transpose(1, 3)
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 3)

        return x


class VggishBottleneck(nn.Module):
    def __init__(self, classes_num):
        super(VggishBottleneck, self).__init__()

        self.conv_block1 = VggishConvBlock(in_channels=1, out_channels=32)
        self.conv_block2 = VggishConvBlock(in_channels=32, out_channels=64)
        self.conv_block3 = VggishConvBlock(in_channels=64, out_channels=128)
        self.conv_block4 = VggishConvBlock(in_channels=128, out_channels=128)

        self.final_conv = nn.Conv2d(in_channels=128, out_channels=classes_num,
                                    kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.final_conv)

    def forward(self, input):
        (_, seq_len, freq_bins) = input.shape

        x = input.view(-1, 1, seq_len, freq_bins)
        '''(samples_num, feature_maps, time_steps, freq_bins)'''

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        bottleneck = F.sigmoid(self.final_conv(x))
        bottleneck = bottleneck*bottleneck
        '''(samples_num, classes_num, time_steps, freq_bins)'''

        return bottleneck


class build_Unet(nn.Module):
    def __init__(self, classes_num):
        super(build_Unet, self).__init__()

        self.conv_block1 = VggishConvBlock(in_channels=1, out_channels=16)
        self.conv_block2 = VggishConvBlock(in_channels=16, out_channels=32)
        self.conv_block3 = VggishConvBlock(in_channels=32, out_channels=64)
        self.conv_block4 = VggishConvBlock(in_channels=64, out_channels=128)

        self.deconv1 = VggishDeConvBlock(in_channels=128, out_channels=64)
        self.deconv2 = VggishDeConvBlock(in_channels=128, out_channels=32)
        self.deconv3 = VggishDeConvBlock(in_channels=64, out_channels=16)
        # self.deconv4 = VggishDeConvBlock(in_channels=32, out_channels=1)

        self.final_conv = nn.Conv2d(in_channels=32, out_channels=classes_num,
                                    kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0), bias=True)

        self.drop = nn.Dropout2d(p=0.2)
        self.init_weights()

    def init_weights(self):
        init_layer(self.final_conv)




    def forward(self, input):
        (_, seq_len, freq_bins) = input.shape

        x = input.view(-1, 1, seq_len, freq_bins)
        '''(samples_num, feature_maps, time_steps, freq_bins)'''

        x = F.pad(x, (0, 0, 1, 0))
        x = self.conv_block1(x)
        down1 = x
        x = F.avg_pool2d(x, 2, 2)
        x = self.conv_block2(x)
        down2 = x
        x = F.avg_pool2d(x, 2, 2)
        x = self.conv_block3(x)
        down3 = x
        x = F.avg_pool2d(x, 2, 2)
        x = self.conv_block4(x)
        # x = self.drop(x)

        x = torch.cat((self.deconv1(x), down3), 1)
        x = torch.cat((self.deconv2(x), down2), 1)
        x = torch.cat((self.deconv3(x), down1), 1)
        # x = self.deconv4(x)

        x = self.final_conv(x)

        bottleneck = F.sigmoid(x[:, :, :-1, :])
        # bottleneck = bottleneck*bottleneck
        '''(samples_num, classes_num, time_steps, freq_bins)'''

        return bottleneck


class build_HornNet(nn.Module):
    def __init__(self, classes_num):
        super(build_HornNet, self).__init__()
        self.mixup_1 = MixupBlock(1, 64)
        self.conv_block1 = VggishConvBlock(in_channels=1, out_channels=16)
        self.conv_block2 = VggishConvBlock(in_channels=16, out_channels=32)
        self.conv_block3 = VggishConvBlock(in_channels=32, out_channels=64)
        self.conv_block4 = VggishConvBlock(in_channels=64, out_channels=128)
        self.conv_block5 = VggishConvBlock(in_channels=128, out_channels=256)
        self.conv_block6 = VggishConvBlock(in_channels=256, out_channels=256)
        self.conv_block7 = VggishConvBlock(in_channels=256, out_channels=256)

        self.deconv0 = VggishDeConvBlock(in_channels=256, out_channels=128, stride=(2, 1))
        self.deconv1 = VggishDeConvBlock(in_channels=256+128, out_channels=128, stride=(2, 1))
        self.deconv2 = VggishDeConvBlock(in_channels=256+128, out_channels=128, stride=(2, 1))
        self.deconv3 = VggishDeConvBlock(in_channels=256, out_channels=64, stride=(2, 1))
        self.deconv4 = VggishDeConvBlock(in_channels=128, out_channels=32, stride=(2, 1))
        self.deconv5 = VggishDeConvBlock(in_channels=64, out_channels=32, stride=(2, 1))
        # self.deconv4 = VggishDeConvBlock(in_channels=32, out_channels=1)

        self.final_conv = nn.Conv2d(in_channels=48, out_channels=classes_num,
                                    kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0), bias=True)

        self.drop = nn.Dropout2d(p=0.2)
        self.init_weights()

    def init_weights(self):
        init_layer(self.final_conv)

    def forward(self, input):
        (_, seq_len, freq_bins) = input.shape
        x = input.view(-1, 1, seq_len, freq_bins)
        '''(samples_num, feature_maps, time_steps, freq_bins)'''

        x = F.pad(x, (0, 0, 1, 0))
        x = self.mixup_1(x)

        x = self.conv_block1(x)
        down1 = x
        x = F.avg_pool2d(x, 2, 2)
        x = self.conv_block2(x)
        down2 = x
        x = F.avg_pool2d(x, 2, 2)
        x = self.conv_block3(x)
        down3 = x
        x = F.avg_pool2d(x, 2, 2)
        x = self.conv_block4(x)
        down4 = x
        x = F.avg_pool2d(x, 2, 2)
        x = self.conv_block5(x)
        down5 = x
        x = F.avg_pool2d(x, 2, 2)
        x = self.conv_block6(x)
        down6 = x
        x = F.avg_pool2d(x, 2, 2)
        x = self.conv_block7(x)

        # x = self.drop(x)
        x = torch.cat((F.pad(self.deconv0(x), (0, 0, 0, 1)), F.avg_pool2d(down6, (1, 2))), 1)
        x = torch.cat((F.pad(self.deconv1(x), (0, 0, 0, 1)), F.avg_pool2d(down5, (1, 4))), 1)
        x = torch.cat((F.pad(self.deconv2(x), (0, 0, 0, 1)), F.avg_pool2d(down4, (1, 8))), 1)
        x = torch.cat((self.deconv3(x), F.avg_pool2d(down3, (1, 16))), 1)
        x = torch.cat((self.deconv4(x), F.avg_pool2d(down2, (1, 32))), 1)
        x = torch.cat((self.deconv5(x), F.avg_pool2d(down1, (1, 64))), 1)
        # x = self.deconv4(x)

        x = self.final_conv(x)

        bottleneck = F.sigmoid(x[:, :, :-1, :])
        # bottleneck = bottleneck*bottleneck
        '''(samples_num, classes_num, time_steps, freq_bins)'''

        return bottleneck


class Vggish(nn.Module):
    def __init__(self, classes_num, pooling='TP', writer=None):
        super(Vggish, self).__init__()
        self.bottleneck = VggishBottleneck(classes_num)

        if pooling == 'TP':
            self.pooling = TP()
        elif pooling == 'GWRP':
            self.pooling = GWRP(311*64)
        elif pooling == 'RGWRP':
            self.pooling = RGWRP(311*64)

    def forward(self, input, return_bottleneck=False):
        bottleneck = self.bottleneck(input)
        output = self.pooling(bottleneck)

        if return_bottleneck:
            return output, bottleneck

        else:
            return output


class Unet(nn.Module):
    def __init__(self, classes_num, pooling='TP', writer=None):

        super(Unet, self).__init__()

        self.bottleneck = build_Unet(classes_num)

        if pooling == 'TP':
            self.pooling = TP()
        elif pooling == 'GWRP':
            self.pooling = GWRP(311 * 64)
            print(pooling)
        elif pooling == 'RGWRP':
            self.pooling = RGWRP(311 * 64)

    def forward(self, input, return_bottleneck=False):
        bottleneck = self.bottleneck(input)
        output = self.pooling(bottleneck)

        if return_bottleneck:
            return output, bottleneck

        else:
            return output


class HornNet(nn.Module):
    def __init__(self, classes_num, pooling='TP', writer=None):
        super(HornNet, self).__init__()
        self.bottleneck = build_HornNet(classes_num)
        self.writer = writer
        self.global_step = 0

        if pooling=='TP':
            self.pooling = TP()
        elif pooling=='GWRP':
            self.pooling = GWRP(311*SQUEEZED_BIN)
            print(pooling)
        elif pooling=='RGWRP':
            self.pooling = RGWRP(311*SQUEEZED_BIN)
        elif pooling=='AM':
            self.pooling = AlphaMEX()
        elif pooling=='M':
            self.pooling = MEX()


    def forward(self, input, return_bottleneck=False):
        bottleneck = self.bottleneck(input)
        output = self.pooling(bottleneck)

        # self.global_step+=1
        # self.writer.add_scalar('alpha', self.pooling.alpha, global_step=self.global_step)

        if return_bottleneck:
            return output, bottleneck

        else:
            return output

