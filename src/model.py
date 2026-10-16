from torch.nn.functional import mse_loss
import torch
import torch.nn as nn
import torch.nn.functional as F


class single_biLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, additional, bidirectional):
        super(single_biLSTM, self).__init__()

        self.additional = additional

        if bidirectional:
            hidden_size2 = hidden_size * 2

        if not bidirectional:
            hidden_size2 = hidden_size

        # 출력을 위한 선형 레이어
        self.additional_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size * 2),
        )

        self.bn = nn.BatchNorm1d(20).double()

        # 첫 번째 LSTM 층
        self.backbone = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                bidirectional=bidirectional, batch_first=True).double()

        # 두 번째 LSTM 층
        self.lstm2 = nn.LSTM(hidden_size2, int(hidden_size / 2), num_layers=num_layers,
                             bidirectional=bidirectional, batch_first=True).double()

        # 세 번째 LSTM 층
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                             bidirectional=bidirectional, batch_first=True).double()

        self.fc = nn.Linear(hidden_size2, output_size).double()

    def forward(self, train, gt=None):
        out, _ = self.backbone(train)
        # out = self.bn(out)
        # out = F.relu(out)

        # x, _ = self.lstm2(out)
        # x = self.bn(x)
        # out, _ = self.lstm3(x)

        if self.additional:
            out = self.additional_layer(out)

        output = self.fc(out)
        # output = self.bn(output)
        # output = F.elu(output)

        output = output[:, -1, :]

        output = output.squeeze()

        if gt != None:
            gt = gt.squeeze()
            loss = torch.sqrt(mse_loss(output, gt))
            return output, loss

        return output


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.MLP = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.fc = nn.Linear(hidden_size, output_size),

    def forward(self, train, gt=None):
        output = self.MLP(train)
        output = output[:, -1, :]
        output = output.squeeze()

        if gt != None:
            gt = gt.squeeze()
            loss = torch.sqrt(mse_loss(output, gt))
            return output, loss

        return output
