import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.parameter import Parameter

from spikingjelly.clock_driven import neuron, layer


from .blocks import SeparableSEWResBlock



###############
### SCALING ###
###############

class MultiplyBy(nn.Module):

    def __init__(self, scale_value: float = 5., learnable: bool = False) -> None:
        super(MultiplyBy, self).__init__()

        if learnable:
            self.scale_value = Parameter(Tensor([scale_value]))
        else:
            self.scale_value = scale_value

    def forward(self, input: Tensor) -> Tensor:
        return torch.mul(input, self.scale_value)




##############
### BLOCKS ###
##############

class PoolingEncoder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size:int, multiply_factor: float = 5.):

        super().__init__()

        self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels = in_channels, out_channels = in_channels, kernel_size = (5, kernel_size, kernel_size), padding = (0, kernel_size//2, kernel_size//2), padding_mode = 'replicate', bias = False, groups = in_channels),
                nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False),
                nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2), padding = 0),
#                nn.BatchNorm3d(64),
                MultiplyBy(multiply_factor),
                neuron.IFNode(),
#                layer.Dropout(p=0.2),
                )

    def forward(self, x):

        return self.conv1(x)



class separable_decoder_block(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, up_size: tuple, bias = False, multiply_factor: float = 10.):

        super(separable_decoder_block, self).__init__()

        self.deconv_1 = nn.Sequential(
            nn.UpsamplingNearest2d(size=(up_size[0] + (kernel_size - 1), up_size[1] + (kernel_size - 1))),
            nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = kernel_size, bias = bias, groups = in_channels),
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = bias),
#            nn.BatchNorm2d(out_channels),
            MultiplyBy(multiply_factor),
            neuron.IFNode(),
#            DropInOut(p_in = 0.01),
        )

    def forward(self, x):

        return self.deconv_1(x)


class separable_predictor_block(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, up_size: tuple, bias = False):

        super(separable_predictor_block, self).__init__()

        self.pred_1 = nn.Sequential(
            nn.UpsamplingNearest2d(size=(up_size[0] + (kernel_size - 1), up_size[1] + (kernel_size - 1))),
            nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = kernel_size, bias = bias, groups = in_channels),
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = bias),
        )

    def forward(self, x):

        return self.pred_1(x)





###############
### NETWORK ###
###############

class NeuronPool_Separable_Pool3d(nn.Module):

    def __init__(self, multiply_factor: float = 10.):

        super().__init__()

        print("USING SPIKING MODEL")

        self.bottom = nn.Sequential(
                nn.Conv3d(in_channels = 2, out_channels = 32, kernel_size = (5, 7, 7), stride = 1, padding = (0, 3, 3), bias = False, padding_mode = 'replicate'),
#                nn.BatchNorm3d(32),
                MultiplyBy(multiply_factor),
                neuron.IFNode(),
                #layer.Dropout(p=0.2),
                )

        self.conv1 = PoolingEncoder(in_channels = 32, out_channels = 64, kernel_size = 7, multiply_factor = multiply_factor)
        self.conv2 = PoolingEncoder(in_channels = 64, out_channels = 128, kernel_size = 7, multiply_factor = multiply_factor)
        self.conv3 = PoolingEncoder(in_channels = 128, out_channels = 256, kernel_size = 7, multiply_factor = multiply_factor)
        self.conv4 = PoolingEncoder(in_channels = 256, out_channels = 512, kernel_size = 7, multiply_factor = multiply_factor)


        # Residual Block

#        self.res1 = separable_residual_block(channels = 512, multiply_factor = multiply_factor)
#        self.res2 = separable_residual_block(channels = 512, multiply_factor = multiply_factor)

        self.res1 = SeparableSEWResBlock(in_channels = 512, multiply_factor = multiply_factor, kernel_size = 7)

        """
        self.res1 = nn.Sequential(
                nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = False, groups = 512),
                nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 1, bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                )
        """

        # Decoder Block

        self.deconv4 = separable_decoder_block(in_channels = 512, out_channels = 256, up_size = (32, 43),  kernel_size = 7, multiply_factor = multiply_factor)
        self.deconv3 = separable_decoder_block(in_channels = 256*2, out_channels = 128, up_size = (65, 86),  kernel_size = 7, multiply_factor = multiply_factor)
        self.deconv2 = separable_decoder_block(in_channels = 128*2, out_channels = 64, up_size = (130, 173),  kernel_size = 7, multiply_factor = multiply_factor)
        self.deconv1 = separable_decoder_block(in_channels = 64*2, out_channels = 32, up_size = (260, 346),  kernel_size = 7, multiply_factor = multiply_factor)

        # Predictor Block


        self.pred4 = separable_predictor_block(in_channels = 256*2, out_channels = 2, up_size = (260, 346), kernel_size = 7)
        self.pred3 = separable_predictor_block(in_channels = 128*2, out_channels = 2, up_size = (260, 346), kernel_size = 7)
        self.pred2 = separable_predictor_block(in_channels = 64*2, out_channels = 2, up_size = (260, 346), kernel_size = 7)
        self.pred1 = separable_predictor_block(in_channels = 32*2, out_channels = 2, up_size = (260, 346), kernel_size = 7)

        self.pool = neuron.IFNode(v_threshold = float('inf'), v_reset = 0.)



    def forward(self, x):

        in_conv1 = self.bottom(x)


        out_conv1 = self.conv1(in_conv1)

        out_conv2 = self.conv2(out_conv1)

        out_conv3 = self.conv3(out_conv2)

        out_conv4 = self.conv4(out_conv3)

        in_res1 = torch.squeeze(out_conv4, 2)

        out_res1 = self.res1(in_res1)
        
        in_deconv4 = out_res1 + out_conv4[:, :, -1]
        out_deconv4 = self.deconv4(in_deconv4)

        #in_deconv3 = out_deconv4 + out_conv3[:, :, -1]
        in_deconv3 = torch.cat((out_deconv4, out_conv3[:,:,-1]), axis = 1)
        out_deconv3 = self.deconv3(in_deconv3)

        #in_deconv2 = out_deconv3 + out_conv2[:, :, -1]
        in_deconv2 = torch.cat((out_deconv3, out_conv2[:,:,-1]), axis = 1)
        out_deconv2 = self.deconv2(in_deconv2)

        #in_deconv1 = out_deconv2 + out_conv1[:, :, -1] 
        in_deconv1 = torch.cat((out_deconv2, out_conv1[:,:,-1]), axis = 1)
        out_deconv1 = self.deconv1(in_deconv1)



        up_4 = self.pred4(in_deconv3)
        up_3 = self.pred3(in_deconv2)
        up_2 = self.pred2(in_deconv1)
        #up_1 = self.pred1(out_deconv1 + in_conv1[:, :, -1]) # NOT SURE ABOUT THIS LINE
        up_1 = self.pred1(torch.cat((out_deconv1, in_conv1[:, :, -1]), axis = 1)) # NOT SURE ABOUT THIS LINE

        self.pool(up_4)
        pred_4 = self.pool.v
        self.pool(up_3)
        pred_3 = self.pool.v
        self.pool(up_2)
        pred_2 = self.pool.v
        self.pool(up_1)
        pred_1 = self.pool.v


        return [pred_4, pred_3, pred_2, pred_1] 
        #return pred_1




