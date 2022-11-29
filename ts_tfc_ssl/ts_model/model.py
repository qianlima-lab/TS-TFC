import torch
import torch.nn as nn


class SqueezeChannels(nn.Module):
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class FCN(nn.Module):
    def __init__(self, num_classes, input_size=1):
        super(FCN, self).__init__()

        self.num_classes = num_classes
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=8, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.network = nn.Sequential(
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            nn.AdaptiveAvgPool1d(1),
            SqueezeChannels(),
        )

    def forward(self, x, vis=False):
        if vis:
            with torch.no_grad():
                vis_out = self.conv_block1(x)
                vis_out = self.conv_block2(vis_out)
                vis_out = self.conv_block3(vis_out)
                return self.network(x), vis_out

        # vis_out = self.conv_block1(x)
        # vis_out = self.conv_block2(vis_out)
        # print("vis_out2.shape = ", vis_out.shape)
        # vis_out = self.conv_block3(vis_out)
        # print("vis_out3.shape = ", vis_out.shape)

        return self.network(x)


class Classifier(nn.Module):
    def __init__(self, input_dims, output_dims) -> None:
        super(Classifier, self).__init__()

        self.dense = nn.Linear(input_dims, output_dims)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.dense(x))


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, embedding_dim=64, output_dim=32) -> None:
        super(ProjectionHead, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, output_dim),
        )

    def forward(self, x):
        return self.projection_head(x)


class base_Model(nn.Module):
    def __init__(self, input_channels=1, kernel_size=25, stride=3, final_out_channels=128, dropout=0.35):
        ## HAR input_channels = 9, kernel_size=8, stride=1, final_out_channels=128, features_len=18
        ## pFD, input_channels=1, kernel_size=32, stride=4, final_out_channels=128, features_len=162
        ## SleepEDF input_channels=1, kernel_size=25, stride=3, final_out_channels=128, features_len=127
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # model_output_dim = configs.features_len
        # self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # print("x.shape = ", x.shape)

        x_flat = x.reshape(x.shape[0], -1)
        # print("x_flat.shape = ", x_flat.shape)
        # logits = self.logits(x_flat)
        return x_flat
