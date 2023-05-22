import torch
from torch import nn

class MultiTaskHead(torch.nn.Module):
    def __init__(self, num_classes=20):
        super(MultiTaskHead, self).__init__()
        self.avg_pool =  nn.AdaptiveAvgPool2d(1)
        self.gender_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )

        self.gender = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=256, out_features=1)
        )

        self.pov_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )

        self.pov = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=256, out_features=3)
        )

        self.sleeve_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )

        self.sleeve = nn.Linear(in_features=256, out_features=2)

        self.label = nn.Linear(in_features=64, out_features=num_classes)

        self.name = "MultiTaskHead"

    def forward(self, x):
        pool_out = self.avg_pool(x)
        pool_out = pool_out.squeeze(axis=(2,3))
        label_out = self.label(pool_out)

        gender_out = self.gender_conv(x)
        gender_out = self.avg_pool(gender_out)
        gender_out = gender_out.squeeze(axis=(2, 3))
        gender_out = self.gender(gender_out)

        pov_out = self.pov_conv(x)
        pov_out = self.avg_pool(pov_out)
        pov_out = pov_out.squeeze(axis=(2, 3))
        pov_out = self.pov(pov_out)

        sleeve_out = self.sleeve_conv(x)
        sleeve_out = self.avg_pool(sleeve_out)
        sleeve_out = sleeve_out.squeeze(axis=(2, 3))
        sleeve_out = self.sleeve(sleeve_out)

        return gender_out, pov_out, sleeve_out, label_out

    def freeze_labels_head(self):
        for param in self.label.parameters(): param.requires_grad = False

    def freeze_pov_head(self):
        for param in self.pov.parameters(): param.requires_grad = False

    def change_num_classes(self, num_classes):
        self.head[4] = nn.Linear(in_features=32, out_features=num_classes)

class OnlyGender(torch.nn.Module):
    def __init__(self, load=False, path=None):
        super(OnlyGender, self).__init__()
        if load:
            self.backbone = torch.load(path)
        else:
            self.backbone = yolox_nano().backbone.backbone

        self.conv = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=256, out_features=1)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.name = "Only Gender"

    def forward(self, x):
        x = self.backbone(x)[self.backbone.out_features[0]]
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.squeeze(axis=(2, 3))
        x = self.fc(x)
        return x

    def freeze_backbone(self):
        for param in self.backbone.parameters(): param.requires_grad = False

    def change_num_classes(self, num_classes):
        self.head[4] = nn.Linear(in_features=32, out_features=num_classes)