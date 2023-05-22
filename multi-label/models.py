import torch
from torch import nn

from yolox.models import yolox_nano


# FIRST MODEL
class MultiLabel(torch.nn.Module):
    def __init__(self, load=False, path=None, num_classes=26):
        super(MultiLabel, self).__init__()
        if load:
            self.backbone = torch.load(path)
        else:
            self.backbone = yolox_nano().backbone.backbone

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=10240, out_features=520),
            nn.Linear(in_features=520, out_features=num_classes)
        )

        self.sigm = nn.Sigmoid()
        self.name = "Flatten+2Lin"
    def forward(self, x):
        x = self.backbone(x)[self.backbone.out_features[2]]
        x = self.head(x)
        x = self.sigm(x)
        return x

    def freeze_backbone(self):
        for param in self.backbone.parameters(): param.requires_grad = False


# MODEL WITH AVG POOLING
class AvgPooling(torch.nn.Module):
    def __init__(self, load=False, path=None, num_classes=26):
        super(AvgPooling, self).__init__()
        if load:
            self.backbone = torch.load(path)
        else:
            self.backbone = yolox_nano().backbone.backbone

        self.avg_pool =  nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=num_classes)
        )

        self.name = "AvgPooling+3Lin+ReLu+DropOut"

    def forward(self, x):
        x = self.backbone(x)[self.backbone.out_features[2]]
        x = self.avg_pool(x)
        x = x.squeeze(axis=(2,3))
        x = self.head(x)
        return x

    def freeze_backbone(self):
        for param in self.backbone.parameters(): param.requires_grad = False

    def change_num_classes(self, num_classes):
        self.head[4] = nn.Linear(in_features=32, out_features=num_classes)

class BatchNorm(torch.nn.Module):
    def __init__(self, load=False, path=None, num_classes=26):
        super(BatchNorm, self).__init__()
        if load:
            self.backbone = torch.load(path)
        else:
            self.backbone = yolox_nano().backbone.backbone

        self.batch_norm = nn.BatchNorm2d(256)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=num_classes)
        )

        self.name = "BatchNorm+AvgPooling+3Lin+ReLu+DropOut"

    def forward(self, x):
        x = self.backbone(x)[self.backbone.out_features[2]]
        x = self.batch_norm(x)
        x = self.avg_pool(x)
        x = x.squeeze(axis=(2, 3))
        x = self.head(x)
        return x

    def freeze_backbone(self):
        for param in self.backbone.parameters(): param.requires_grad = False

    def change_num_classes(self, num_classes):
        self.head[4] = nn.Linear(in_features=32, out_features=num_classes)

class MultiTask(torch.nn.Module):
    def __init__(self, load=False, path=None, num_classes=20):
        super(MultiTask, self).__init__()
        if load:
            self.backbone = torch.load(path)
        else:
            self.backbone = yolox_nano().backbone.backbone

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

        self.name = "MultiTaskV3"

    def forward(self, x):
        back_out = self.backbone(x)[self.backbone.out_features[0]]
        pool_out = self.avg_pool(back_out)
        pool_out = pool_out.squeeze(axis=(2,3))
        label_out = self.label(pool_out)

        gender_out = self.gender_conv(back_out)
        gender_out = self.avg_pool(gender_out)
        gender_out = gender_out.squeeze(axis=(2, 3))
        gender_out = self.gender(gender_out)

        pov_out = self.pov_conv(back_out)
        pov_out = self.avg_pool(pov_out)
        pov_out = pov_out.squeeze(axis=(2, 3))
        pov_out = self.pov(pov_out)

        sleeve_out = self.sleeve_conv(back_out)
        sleeve_out = self.avg_pool(sleeve_out)
        sleeve_out = sleeve_out.squeeze(axis=(2, 3))
        sleeve_out = self.sleeve(sleeve_out)

        return gender_out, pov_out, sleeve_out, label_out

    def freeze_backbone(self):
        for param in self.backbone.parameters(): param.requires_grad = False

    def freeze_labels_head(self):
        for param in self.label.parameters(): param.requires_grad = False

    def freeze_pov_head(self):
        for param in self.pov.parameters(): param.requires_grad = False

    def change_num_classes(self, num_classes):
        self.head[4] = nn.Linear(in_features=32, out_features=num_classes)

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