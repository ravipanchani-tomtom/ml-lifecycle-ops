from torch import nn
from torch.functional import F
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.AdaptiveAvgPool2d((1,1))
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.fc1(x))
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f'{name}: {param.numel()} parameters')


class DynamicMNISTModel(nn.Module):
    def __init__(self, num_classes=10, num_blocks=3, initial_channels=8):
        super(DynamicMNISTModel, self).__init__()
        self.num_blocks = num_blocks
        self.initial_channels = initial_channels

        layers = []
        in_channels = 1
        for i in range(num_blocks):
            out_channels = initial_channels * (2 ** i)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, num_classes)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f'{name}: {param.numel()} parameters')


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Encoder path (increasing channels)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Decoder path (decreasing channels)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 8, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(8 * 3 * 3, 32)    # Adjusted for 8 channels
        self.fc2 = nn.Linear(32, 32)           # Middle layer
        self.fc3 = nn.Linear(32, 10)           # Output layer

        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Encoder path with pooling
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = self.pool(F.relu(self.conv3(x)))  # 7x7 -> 3x3

        # Decoder path without pooling (reducing channels)
        x = F.relu(self.conv4(x))  # Stays at 3x3, reduces to 16 channels
        x = F.relu(self.conv5(x))  # Stays at 3x3, reduces to 8 channels

        # Flatten and fully connected layers
        x = x.view(-1, 8 * 3 * 3)  # Now using 8 channels
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f'{name}: {param.numel()} parameters')