from torch import nn
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Normalize

import torch.nn.functional as F
import torch
import torchvision as tv
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "resnet18": (tv.models.resnet18, 512),
    "resnet34": (tv.models.resnet34, 512),
    "resnet50": (tv.models.resnet50, 2048),
    "resnet101": (tv.models.resnet101, 2048),
    "resnet152": (tv.models.resnet152, 2048),
}

def swish(x):
    return x * F.sigmoid(x)

# def aestheic_transform():
#     return Compose([
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        # y = swish(self.conv1(x))
        return self.bn2(self.conv2(y)) + x

        # return self.conv2(y) + x


class upsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return swish(self.shuffler(self.conv(x)))


class SingleGenerator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(SingleGenerator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(12, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(self.upsample_factor//2):
            self.add_module('upsample' + str(i+1), upsampleBlock(64, 256))

        self.conv3 = nn.Conv2d(64, 12, 9, stride=1, padding=4)

    def forward(self, x):
        x = swish(self.conv1(x))

        y_old = x.clone()

        y = y_old
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(self.upsample_factor//2):
            x = self.__getattr__('upsample' + str(i+1))(x)

        return self.conv3(x)


class SingleDiscriminator(nn.Module):
    def __init__(self):
        super(SingleDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)

    def forward(self, x):
        x = swish(self.conv1(x))

        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        x = swish(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)


class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            *list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)

# Modelv3 with skip connections


class MultipleGenerator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(MultipleGenerator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1_in1 = nn.Conv2d(3, 64, 9, stride=1, padding=4)
        self.conv1_in2 = nn.Conv2d(3, 64, 9, stride=1, padding=4)
        self.conv1_in3 = nn.Conv2d(3, 64, 9, stride=1, padding=4)
        self.conv1_in4 = nn.Conv2d(3, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.bn7 = nn.BatchNorm2d(64)

        self.conv8 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.conv6 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(64)

        for i in range(self.upsample_factor//2):
            self.add_module('upsample' + str(i+1), upsampleBlock(64, 256))

        self.conv5_out1 = nn.Conv2d(64, 3, 9, stride=1, padding=4)
        self.conv5_out2 = nn.Conv2d(64, 3, 9, stride=1, padding=4)
        self.conv5_out3 = nn.Conv2d(64, 3, 9, stride=1, padding=4)
        self.conv5_out4 = nn.Conv2d(64, 3, 9, stride=1, padding=4)

    def forward(self, x1, x2, x3, x4):
        x1 = swish(self.conv1_in1(x1))
        x2 = swish(self.conv1_in1(x2))
        x3 = swish(self.conv1_in1(x3))
        x4 = swish(self.conv1_in1(x4))

        y_old = torch.cat([x1.clone(), x2.clone(), x3.clone(), x4.clone()], 0)
        y = y_old

        y = self.conv7(y)

        y = self.conv8(y) + y_old

        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)

        x = self.conv2(y)
        x = self.conv6(x)
        x = self.conv3(x)
        x = self.conv4(x) + y_old

        for i in range(self.upsample_factor//2):
            x = self.__getattr__('upsample' + str(i+1))(x)

        x = torch.chunk(x, 4, 0)
        # import ipdb
        # ipdb.set_trace()
        return torch.cat([self.conv5_out1(x[0]), self.conv5_out2(x[1]), self.conv5_out3(x[2]), self.conv5_out3(x[3])], 0)


class MultipleDiscriminator(nn.Module):
    def __init__(self):
        super(MultipleDiscriminator, self).__init__()
        self.conv_in1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_in2 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_in3 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_in4 = nn.Conv2d(3, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # Replaced original paper FC layers with FCN
        self.conv_out9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)
        # self.conv_out10 = nn.Conv2d(512, 1, 1, stride=1, padding=1)
        # self.conv_out11 = nn.Conv2d(512, 1, 1, stride=1, padding=1)
        # self.conv_out12 = nn.Conv2d(512, 1, 1, stride=1, padding=1)

    def forward(self, x1, x2, x3, x4):
        x1 = swish(self.conv_in1(x1))
        x2 = swish(self.conv_in1(x2))
        x3 = swish(self.conv_in1(x3))
        x4 = swish(self.conv_in1(x4))

        x = torch.cat([x1, x2, x3, x4], 0)
        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        x = swish(self.bn8(self.conv8(x)))

        x = self.conv_out9(x)
        # x = self.conv9(x2)
        # x = self.conv9(x3)
        # x = self.conv9(x4)
        return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)

# new ang res model
class AngRes(nn.Module):
    def __init__(self):
        super(AngRes, self).__init__()
        self.conv_in1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv_in2 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv_in3 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv_in4 = nn.Conv2d(3, 32, 3, stride=1, padding=1)

        self.conv_l1 = nn.Conv2d(128, 256, 5, stride=1, padding=2)
        self.conv_l2 = nn.Conv2d(256, 512, 5, stride=1, padding=2)
        self.conv_l3 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.conv_l4 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv_l5 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv_l6 = nn.Conv2d(512, 512, 5, stride=1, padding=2)
        self.conv_l7 = nn.Conv2d(512, 256, 5, stride=1, padding=2)

        self.conv_out1 = nn.Conv2d(32, 3, 3, stride=1, padding=1)
        self.conv_out2 = nn.Conv2d(32, 3, 3, stride=1, padding=1)
        self.conv_out3 = nn.Conv2d(32, 3, 3, stride=1, padding=1)
        self.conv_out4 = nn.Conv2d(32, 3, 3, stride=1, padding=1)
        self.conv_out5 = nn.Conv2d(32, 3, 3, stride=1, padding=1)

    def forward(self, x1, x2, x3, x4):
        x1 = swish(self.conv_in1(x1))
        x2 = swish(self.conv_in2(x2))
        x3 = swish(self.conv_in2(x3))
        x4 = swish(self.conv_in2(x4))

        y = torch.cat([x1.clone(), x2.clone(), x3.clone(), x4.clone()], 1)

        y = swish(self.conv_l1(y))
        y = swish(self.conv_l2(y))
        y = swish(self.conv_l6(y))
        y = swish(self.conv_l7(y))
        y = swish(self.conv_l3(y))
        y = swish(self.conv_l4(y))
        y = swish(self.conv_l5(y))
        
        y1 = swish(self.conv_out1(y))
        y2 = swish(self.conv_out2(y))
        y3 = swish(self.conv_out3(y))
        y4 = swish(self.conv_out4(y))
        y5 = swish(self.conv_out5(y))
        return torch.cat([y1, y2, y3, y4, y5], 0)

class InferAesthetic(nn.Module):
    def __init__(self):
        super(InferAesthetic, self).__init__()
        # self.transform = aesthetic_transform()
        model_state = torch.load(
            "checkpoints/NIMA.pth", map_location=lambda storage, loc: storage)
        self.model = create_model(
            model_type=model_state["model_type"], drop_out=0)
        self.model.load_state_dict(model_state["state_dict"])
        self.model = self.model.to(DEVICE)
        # self.model.eval()

    def get_mean_score(self, score):
        buckets = torch.arange(1, 11).to(DEVICE)
        mu = torch.sum(buckets * score)
        return mu

    def forward(self, image):
        with torch.no_grad():
            # image = self.transform(image)
            # time.sleep(5)
            prob = self.model(image.unsqueeze_(0))
            mean_score = self.get_mean_score(prob)

            # Scales mean score to between 0-1, as opposed to 1-10
            mean_score_normalised = (mean_score - 1) / 9

            return mean_score_normalised


class NIMA(nn.Module):
    def __init__(self, base_model: nn.Module, input_features: int, drop_out: float):
        super(NIMA, self).__init__()
        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True), nn.Dropout(p=drop_out), nn.Linear(
                input_features, 10), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


def create_model(model_type: str, drop_out: float) -> NIMA:
    create_function, input_features = MODELS[model_type]
    base_model = create_function(pretrained=True)
    base_model = nn.Sequential(*list(base_model.children())[:-1])
    return NIMA(base_model=base_model, input_features=input_features, drop_out=drop_out)

