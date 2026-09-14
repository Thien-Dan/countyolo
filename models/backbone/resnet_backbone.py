import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Backbone(nn.Module):
    """
    ResNet-50 Backbone lấy từ torchvision.
    Chỉ lấy các layer trích xuất đặc trưng (lên đến layer4), bỏ qua fully connected layer.
    """
    def __init__(self, in_channels=3, out_channels=256, pretrained=True):
        super().__init__()
        
        # Tải mô hình ResNet-50
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)
        
        # Nếu in_channels != 3, điều chỉnh lớp conv1 (Mặc định là 3 nên thường không cần)
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        # Lấy các khối trích xuất đặc trưng
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4 # Output channel = 2048
        
        # Giảm chiều channel từ 2048 xuống `out_channels` (mặc định 256) cho phù hợp với CountYOLO Neck
        self.reduction_conv = nn.Conv2d(2048, out_channels, kernel_size=1, bias=False)
        self.reduction_bn = nn.BatchNorm2d(out_channels)
        self.reduction_act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Giảm chiều channel
        x = self.reduction_conv(x)
        x = self.reduction_bn(x)
        x = self.reduction_act(x)
        
        # Upsample 8x để đưa feature map từ stride 32 (layer4) về stride 4 (như kỳ vọng của DummyNeck/Head)
        x = nn.functional.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        
        return x
