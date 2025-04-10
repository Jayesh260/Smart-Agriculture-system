import torch
import torch.nn as nn

# Define ConvBlock and ResNet9 here (same as in training)
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_diseases)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load full model with class present
full_model = torch.load(r'D:\NewCode\Allcombined\DiseaseNew\model\plant-disease-model-complete.pth', map_location=device)

# Create same architecture
model = ResNet9(in_channels=3, num_diseases=38)  # Replace 38 if you have a different number of classes
model.load_state_dict(full_model.state_dict())

# Save only the state_dict (safe way)
torch.save(model.state_dict(), r'D:\NewCode\Allcombined\DiseaseNew\model\plant-disease-model-weights.pth')

print("✅ Model resaved with state_dict successfully.")
