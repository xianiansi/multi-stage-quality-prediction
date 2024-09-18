####################新数据集seq2seq###################
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm
import torch
from torch import nn
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
import warnings
import torch.nn.functional as F
from models import *
warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 读取数据
df = pd.read_csv("manufacturing.csv")
y_1 = df["quality"][:6000].values.reshape(-1, 1)
y_2 = df["quality"][:6000].values.reshape(-1, 1)
df = pd.read_csv("data_Y.csv")
y_3 = df["quality"][:6000].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaler.fit(y_3)
y_3_scaled = scaler.transform(y_3)
y_3 = y_3_scaled.flatten()
y_array = np.array(y_1*y_2*y_3)


df = pd.read_csv("manufacturing.csv")
x1 = df.iloc[:3957, :-1]
df = pd.read_excel("quality.xlsx")
x2 = df.iloc[:3957, :-1]
df = pd.read_csv("data_X.csv")
x3 = df.iloc[:3957, 1:6]

# Convert DataFrames to NumPy arrays and add a new dimension at axis 1
x1_array = np.expand_dims(x1.to_numpy(), axis=1)
x2_array = np.expand_dims(x2.to_numpy(), axis=1)
x3_array = np.expand_dims(x3.to_numpy(), axis=1)

# Concatenate the arrays along the new dimension (axis 1)
x_array = np.concatenate((x1_array, x2_array, x3_array), axis=1)
# scaler = StandardScaler()
# x_array_standardized = np.array([scaler.fit_transform(sample) for sample in x_array])
x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.2, random_state=42)

# 将三维数组展平为二维数组
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
#
#
# # 定义Dataset和DataLoader
# class DataSet(Data.Dataset):
#     def __init__(self, data_inputs, data_targets):
#         self.inputs = torch.FloatTensor(data_inputs)
#         self.label = torch.FloatTensor(data_targets)
#
#     def __getitem__(self, index):
#         return self.inputs[index], self.label[index]
#
#     def __len__(self):
#         return len(self.inputs)
#
#
# # Batch_Size = 128
# train_dataset = DataSet(x_train, y_train)
# test_dataset = DataSet(x_test, y_test)
#
# TrainDataLoader = Data.DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True, drop_last=True)
# TestDataLoader = Data.DataLoader(test_dataset, batch_size=Batch_Size, shuffle=True, drop_last=True)
#
# # 定义模型、损失函数和优化器
# model = CNNLSTMAttention(input_size=5, hidden_size=128, num_layers=2, output_size=1).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
# criterion = nn.MSELoss().to(device)
#
#
# def test(loader):
#     model.eval()
#     with torch.no_grad():
#         val_epoch_loss = []
#         for inputs, targets in loader:
#             inputs, targets = inputs.to(device).float(), targets.to(device).float().view(-1, 1)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             val_epoch_loss.append(loss.item())
#         return np.mean(val_epoch_loss)
#
#
# # 训练模型
# epochs = 30
# test_loss = []
# train_loss = []
# best_test_loss = float('inf')
#
# for epoch in tqdm(range(epochs)):
#     model.train()
#     train_epoch_loss = []
#     for inputs, targets in TrainDataLoader:
#         inputs, targets = inputs.to(device).float(), targets.to(device).float().view(-1, 1)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         train_epoch_loss.append(loss.item())
#     train_loss.append(np.mean(train_epoch_loss))
#
#     test_epoch_loss = test(TestDataLoader)
#     test_loss.append(test_epoch_loss)
#
#     print(f"Epoch: {epoch + 1}, Train Loss: {np.mean(train_epoch_loss)}, Test Loss: {test_epoch_loss}")
#
# # 画损失图
# fig = plt.figure(facecolor='white', figsize=(10, 7))
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.plot(range(len(train_loss)), train_loss, label='Train Loss')
# plt.plot(range(len(test_loss)), test_loss, label='Test Loss')
# plt.title('Loss based on CNNLSTMAttention')
# plt.legend()
# plt.show()
#
# # 加载最佳模型并进行预测
# # model.load_state_dict(torch.load('best_LSTM_model.pth'))
# model.eval()
#
# y_pred = []
# y_true = []
# with torch.no_grad():
#     for inputs, targets in TestDataLoader:
#         inputs, targets = inputs.to(device).float(), targets.to(device).float()
#         outputs = model(inputs)
#         y_pred.extend(outputs.squeeze().cpu().numpy())
#         y_true.extend(targets.cpu().numpy())



# XGBoost 回归器
xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)


# 画折线图显示
plt.plot(y_test, label='True Values', marker='o')
plt.plot(y_pred, label='Predicted Values', marker='x')
plt.xlabel('Samples')
plt.ylabel('Qualified Rate')
plt.title('Qualified Rate based on CNNLSTMAttention')
plt.legend()
plt.show()

# 计算评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_true = y_test
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"均方误差 (MSE): {mse}")
print(f"均方根误差 (RMSE): {rmse}")
print(f"平均绝对误差 (MAE): {mae}")
print(f"决定系数 (R²): {r2}")


