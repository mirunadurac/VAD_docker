import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)


    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        return out[:, -1, :]

class DNN(nn.Module):
    def __init__(self,input_dim, dim_mid,output_dim):
        super(DNN, self).__init__()

        self.fc1 = nn.Sequential(
            torch.nn.Linear(input_dim, dim_mid),
            # torch.nn.BatchNorm1d(dim_mid)
        )

        self.fc2 = nn.Sequential(
            torch.nn.Linear(dim_mid, dim_mid),
            # torch.nn.BatchNorm1d(dim_mid)

        )
        self.fc3 = nn.Sequential(
            torch.nn.Linear(dim_mid, output_dim),
            # torch.nn.BatchNorm1d(64)
        )

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))

        return x



class CLDNNModel(nn.Module):
    def __init__(self, input_dim=0, hidden_dim=0, layer_dim=0, output_dim=0, dim_mid=0):
        super(CLDNNModel, self).__init__()

        self.conv1 = nn.Conv1d(10, out_channels=256,kernel_size=9)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=4)
        self.p1=nn.MaxPool1d(kernel_size=3)
        self.LSTM=LSTM(7, 32, 3)
        self.DNN=DNN(32,32, 2)

    def forward(self, x):
        out=self.conv1(x)
        out=self.p1(out)
        out=torch.relu(out)
        out=self.conv2(out)
        out = torch.relu(out)
        out=self.LSTM(out)
        out=self.DNN(out)

        return out