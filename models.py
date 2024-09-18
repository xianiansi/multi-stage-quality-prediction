import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size,  hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class LSTMDropout(nn.Module):
    def __init__(self, input_size,  hidden_size, num_layers, output_size):
        super(LSTMDropout, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Get the last time step output
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        return out

# Seq2Seq Model
class Seq2SeqModel(nn.Module):
    def __init__(self,input_size,  hidden_size, num_layers, output_size):
        super(Seq2SeqModel, self).__init__()
        # 输入：(batch_size, seq_length, input_size)
        # self.lstm1输出：(batch_size, seq_length, hidden_size)但是取最后一个时间步的数据(batch_size, hidden_size)
        # self.fc1输出：(batch_size, 10)
        # Repeat Vector输出: (batch_size, seq_length, 128)
        # LSTM2输出: (batch_size, seq_length, hidden_size)
        # 最终输出: (batch_size, seq_length, output_size)
        # 编码层用lstm预测最后一步输出，前馈网络传播；解码层再广播至每个时间步，lstm输出加前馈网络
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 10)
        self.repeat_vector = nn.Linear(10, 128)  # Placeholder for RepeatVector
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        fc1_out = F.relu(self.fc1(lstm_out1[:, -1, :]))  # Apply Dense layer
        repeat_out = self.repeat_vector(fc1_out).unsqueeze(1).repeat(1, x.size(1), 1)
        lstm_out2, _ = self.lstm2(repeat_out)
        out = self.fc2(lstm_out2[:, -1, :])
        return out

class Seq2SeqModel_GRU(nn.Module):
    def __init__(self,input_size,  hidden_size, num_layers, output_size):
        super(Seq2SeqModel_GRU, self).__init__()
        self.GRU1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                           bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(2 * hidden_size, 10)
        self.repeat_vector = nn.Linear(10, 128)  # Placeholder for RepeatVector
        self.GRU2 = nn.GRU(input_size=128, hidden_size=hidden_size, num_layers=num_layers,
                           bidirectional=True, batch_first=True)
        self.fc2 = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        GRU_out1, _ = self.GRU1(x)
        fc1_out = F.relu(self.fc1(GRU_out1[:, -1, :]))  # Apply Dense layer
        repeat_out = self.repeat_vector(fc1_out).unsqueeze(1).repeat(1, x.size(1), 1)
        GRU_out2, _ = self.GRU2(repeat_out)
        out = self.fc2(GRU_out2[:, -1, :])
        return out

# Seq2Seq with Attention Model
class Seq2Seq_GRU_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Seq2Seq_GRU_Attention, self).__init__()
        self.GRU1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True)
        self.attention = AttentionBlock(hidden_size, 6*hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.repeat_vector = nn.Linear(hidden_size, hidden_size)
        self.GRU2 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        GRU_out1, _ = self.GRU1(x)
        attention_out = self.attention(GRU_out1)
        attention_out = attention_out.view(attention_out.size(0), -1)
        fc1_out = self.fc1(attention_out)
        repeat_out = self.repeat_vector(fc1_out).unsqueeze(1).repeat(1, x.size(1), 1)
        GRU_out2, _ = self.GRU2(repeat_out)
        GRU_out2 = GRU_out2[:,-1,:]
        out = self.fc2(GRU_out2)
        return out

# Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionBlock, self).__init__()
        # 输入： (batch_size, time_step, input_size)
        # Wa输出: (batch_size, time_step, hidden_size)
        # Ws输出: (batch_size, time_step, 1)
        # Softmax后输出: (batch_size, time_step, 1) 每一步的注意力概率
        # 批量矩阵乘法 (torch.bmm) a.permute(0, 2, 1)，形状为 (batch_size, 1, time_step)每一步的注意力
        # x，形状为 (batch_size, time_step, hidden_size) 原来的数据
        # 每个batch都会进行二维矩阵相乘，所以输出: context_vector (batch_size, 1, hidden_size)
        # 上下文向量 (context_vector): (batch_size, 1, hidden_size)
        self.Wa = nn.Linear(input_size, hidden_size)
        self.Ws = nn.Linear(hidden_size, 1)

    def forward(self, x):
        a = self.Wa(x)
        a = torch.tanh(a)
        a = self.Ws(a)
        a = F.softmax(a, dim=1)
        context_vector = torch.bmm(a.permute(0, 2, 1), x)
        return context_vector


# Attention Model
class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(AttentionModel, self).__init__()
        # 输入: (batch_size, time_step, input_size)
        # LSTM输出: (batch_size, time_step, hidden_size)
        # Attention Block输出: (batch_size, 1, hidden_size)
        # view改变形状输出: (batch_size, hidden_size) hidden_size这一维相当于每个时间步的加权求和，权重是注意力给的
        # fc输出: (batch_size, output_size)
        # 可以说是消融掉时间步这个维度了
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers, batch_first=True, dropout=0.8)
        self.attention = AttentionBlock(hidden_size,2*hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_out = self.attention(lstm_out)
        attention_out = attention_out.view(attention_out.size(0), -1)
        out = self.fc(attention_out)
        return out


# Seq2Seq with Attention Model
class Seq2SeqAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Seq2SeqAttention, self).__init__()
        # 输入: (batch_size, time_step, input_size)
        # LSTM1输出: (batch_size, time_step, hidden_size)
        # Attention Block输出: (batch_size, 1, hidden_size)
        # view后的输出: (batch_size, hidden_size)
        # fc1输出: (batch_size, output_size)
        # Repeat Vector输出: (batch_size, time_step, hidden_size)
        # LSTM2输出: (batch_size, time_step, 2*hidden_size)
        # 最终输出: (batch_size, time_step, output_size)
        # 其实就是在Seq2Seq的编码端加上了一个Attention模块，原来的Seq2Seq就是直接取编码端lstm最后一步的输出，再广播至time_step
        # 现在针对每个时间步做了一个加权再输出，再广播、解码
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                             bidirectional=True,batch_first=True)
        self.attention = AttentionBlock(2*hidden_size, 2*hidden_size)
        self.fc1 = nn.Linear(2*hidden_size, output_size)
        self.repeat_vector = nn.Linear(output_size, hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, 2*hidden_size,batch_first=True)
        self.fc2 = nn.Linear(2*hidden_size, output_size)

    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        attention_out = self.attention(lstm_out1)
        attention_out = attention_out.view(attention_out.size(0), -1)
        fc1_out = self.fc1(attention_out)
        repeat_out = self.repeat_vector(fc1_out).unsqueeze(1).repeat(1, x.size(1), 1)
        lstm_out2, _ = self.lstm2(repeat_out)
        lstm_out2 = lstm_out2[:,-1,:]
        out = self.fc2(lstm_out2)
        return out

# Seq2Seq with Attention Model
class Seq2SeqAttention_dropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Seq2SeqAttention_dropout, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                             bidirectional=True,batch_first=True)
        self.attention = AttentionBlock(2*hidden_size, 2*hidden_size)
        self.fc1 = nn.Linear(2*hidden_size, output_size)
        self.repeat_vector = nn.Linear(output_size, hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, 2*hidden_size,batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(2*hidden_size, output_size)

    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        attention_out = self.attention(lstm_out1)
        attention_out = attention_out.view(attention_out.size(0), -1)
        fc1_out = self.fc1(attention_out)
        repeat_out = self.repeat_vector(fc1_out).unsqueeze(1).repeat(1, x.size(1), 1)
        lstm_out2, _ = self.lstm2(repeat_out)
        lstm_out2 = lstm_out2[:,-1,:]
        out = self.dropout(lstm_out2)
        out = self.fc2(out)
        return out

class CNNLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CNNLSTMAttention, self).__init__()
        # 输入: (batch_size, time_step, input_size) x.permute之后(batch_size, input_size, time_step)
        # 卷积层输出: (batch_size, hidden_size, time_step)
        # 再permute: (batch_size, time_step, hidden_size)
        # LSTM 输出: (batch_size, time_step, 2 * hidden_size)
        # 注意力层输出: (batch_size, 1, hidden_size)
        # view展平后的输出: (batch_size, hidden_size)
        # out: (batch_size, output_size)
        # 这里卷积层也是使用多个卷积核对input_size做映射，由于conv层的输入输出通道属性，所以需要permute变换一下形状
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=1)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.3)
        self.attention = AttentionBlock(hidden_size * 2, hidden_size*2)
        self.fc = nn.Linear(hidden_size*2, output_size)  # Adjusted to match flattened size

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, sequence_length)
        x = self.conv1(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, sequence_length, channels)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout2(lstm_out)
        attention_out = self.attention(lstm_out)
        attention_out = attention_out.view(attention_out.size(0), -1)  # Flatten the tensor
        out = self.fc(attention_out)
        return out