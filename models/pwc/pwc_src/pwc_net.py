''' (v-iashin):
    Credits: https://github.com/sniklaus/pytorch-pwc/f6138900578214ab4e3daef6743b88f7824293be
    This is just a wrapper for `run.py` from `pytorch_pwc` repo.
    No refactoring if not stated in comments.
    I need to rewrite it a bit as I cannot import `run.py` without CLI arguments
    but I want to use it for debugging
'''
import models.pwc.pwc_src.correlation as correlation

import numpy
import math
import torch

# (v-iashin) adding these two for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
numpy.random.seed(0)
#

assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 41) # requires at least pytorch version 0.4.1

def Backward(tensorInput, tensorFlow, device):

    Backward_tensorGrid = {}
    Backward_tensorPartial = {}

    tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
    tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

    Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).to(device)
    Backward_tensorPartial[str(tensorFlow.size())] = tensorFlow.new_ones([tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3)])

    tensorFlow = torch.cat([tensorFlow[:, [0], :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, [1], :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)
    tensorInput = torch.cat([ tensorInput, Backward_tensorPartial[str(tensorFlow.size())] ], 1)

    tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')

    tensorMask = tensorOutput[:, -1:, :, :]; tensorMask[tensorMask > 0.999] = 1.0; tensorMask[tensorMask < 1.0] = 0.0

    return tensorOutput[:, :-1, :, :] * tensorMask


class Extractor(torch.nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()

        self.moduleOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThr = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFou = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSix = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

    def forward(self, tensorInput):
        tensorOne = self.moduleOne(tensorInput)
        tensorTwo = self.moduleTwo(tensorOne)
        tensorThr = self.moduleThr(tensorTwo)
        tensorFou = self.moduleFou(tensorThr)
        tensorFiv = self.moduleFiv(tensorFou)
        tensorSix = self.moduleSix(tensorFiv)

        return [ tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix ]

class Decoder(torch.nn.Module):
    def __init__(self, intLevel):
        super(Decoder, self).__init__()

        intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
        intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

        if intLevel < 6: self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 6: self.moduleUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 6: self.dblBackward = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

        self.moduleOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThr = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFou = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSix = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, tensorFirst, tensorSecond, objectPrevious, device):

        tensorFlow = None
        tensorFeat = None

        if objectPrevious is None:
            tensorFlow = None
            tensorFeat = None

            tensorVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tensorFirst, tensorSecond, device),
                                                          negative_slope=0.1, inplace=False)

            tensorFeat = torch.cat([ tensorVolume ], 1)

        elif objectPrevious is not None:
            tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
            tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])
            tensorVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=Backward(tensorInput=tensorSecond,
tensorFlow=tensorFlow * self.dblBackward, device=device), device=device), negative_slope=0.1, inplace=False)

            tensorFeat = torch.cat([ tensorVolume, tensorFirst, tensorFlow, tensorFeat ], 1)

        # end

        tensorFeat = torch.cat([ self.moduleOne(tensorFeat), tensorFeat ], 1)
        tensorFeat = torch.cat([ self.moduleTwo(tensorFeat), tensorFeat ], 1)
        tensorFeat = torch.cat([ self.moduleThr(tensorFeat), tensorFeat ], 1)
        tensorFeat = torch.cat([ self.moduleFou(tensorFeat), tensorFeat ], 1)
        tensorFeat = torch.cat([ self.moduleFiv(tensorFeat), tensorFeat ], 1)

        tensorFlow = self.moduleSix(tensorFeat)

        return {
            'tensorFlow': tensorFlow,
            'tensorFeat': tensorFeat
        }

class Refiner(torch.nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()

        self.moduleMain = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
        )

    def forward(self, tensorInput):
        return self.moduleMain(tensorInput)

class PWCNet(torch.nn.Module):
    def __init__(self):
        super(PWCNet, self).__init__()

        self.moduleExtractor = Extractor()

        self.moduleTwo = Decoder(2)
        self.moduleThr = Decoder(3)
        self.moduleFou = Decoder(4)
        self.moduleFiv = Decoder(5)
        self.moduleSix = Decoder(6)

        self.moduleRefiner = Refiner()

    def forward(self, tensorFirst, tensorSecond):
        device = tensorFirst.device

        # (v-iashin) [2, 1, 0] transforms RGB to BGR -- replicates the [::-1] behaviour in this case
        tensorFirst = tensorFirst[:, [2, 1, 0], :, :] / 255
        tensorSecond = tensorSecond[:, [2, 1, 0], :, :] / 255

        assert(tensorFirst.size(1) == tensorSecond.size(1))
        assert(tensorFirst.size(2) == tensorSecond.size(2))

        T, C, intHeight, intWidth = tensorFirst.size()

        tensorPreprocessedFirst = tensorFirst.view(T, C, intHeight, intWidth)
        tensorPreprocessedSecond = tensorSecond.view(T, C, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

        tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

        tensorPreprocessedFirst = self.moduleExtractor(tensorPreprocessedFirst)
        tensorPreprocessedSecond = self.moduleExtractor(tensorPreprocessedSecond)

        objectEstimate = self.moduleSix(tensorPreprocessedFirst[-1], tensorPreprocessedSecond[-1], None, device)
        objectEstimate = self.moduleFiv(tensorPreprocessedFirst[-2], tensorPreprocessedSecond[-2], objectEstimate, device)
        objectEstimate = self.moduleFou(tensorPreprocessedFirst[-3], tensorPreprocessedSecond[-3], objectEstimate, device)
        objectEstimate = self.moduleThr(tensorPreprocessedFirst[-4], tensorPreprocessedSecond[-4], objectEstimate, device)
        objectEstimate = self.moduleTwo(tensorPreprocessedFirst[-5], tensorPreprocessedSecond[-5], objectEstimate, device)

        temp = objectEstimate['tensorFlow'] + self.moduleRefiner(objectEstimate['tensorFeat'])

        tensorFlow = 20.0 * torch.nn.functional.interpolate(input=temp, size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tensorFlow
