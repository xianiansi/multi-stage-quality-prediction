# import pandas as pd
#
# # 读取数据
# df = pd.read_csv("manufacturing.csv")
# df2 = pd.read_excel("quality.xlsx")
#
# # 计算data_X和data_y
# data_X = df[["Temperature", "Pressure", "Temperature_x_Pressure", "Material_Fusion_Metric", "Material_Transformation_Metric"]].add(
#     df2[["Temperature", "Humidity", "Proficiency", "Usage_Time", "fixture_accuracy"]], fill_value=0)
#
# data_y = df['Quality_Rating'] * df2['Qualified_rate']
#
# # 合并data_X和data_y
# combined_data = pd.concat([data_X, data_y], axis=1)
#
# # 重命名data_y列
# combined_data.columns = list(data_X.columns) + ['Quality_Rating_Adjusted']
#
# # 保存到新的CSV文件
# combined_data.to_csv("combined_data.csv", index=False)


##################多项式回归,没用train_test###########################
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 两两关系分布
# df=pd.read_csv("manufacturing.csv")
# sns.pairplot(data=df)
# plt.show()
#
# # 分成x和y
# x=df.iloc[:,:-1]
# y=df["Quality_Rating"]
# # 每一列绘制箱型图
# for i, col in enumerate(x.columns):
#     plt.subplot(2, 3, i+1)
#     sns.boxplot(y=col, data=df)
#     plt.tight_layout()
# plt.show()
#
#
# from sklearn.preprocessing import StandardScaler
# ss=StandardScaler()
# ss.fit(x)
# X=pd.DataFrame(ss.transform(x),columns=x.columns)
# sns.heatmap(data=df.corr(),annot=True)
# # plt.xticks(rotation=45, ha='right')
# plt.show()
#
# # 初始化多项式特征生成器 pf，将特征数据扩展到二次项。
# # pf.fit(X) 根据标准化后的特征数据 X 学习数据的结构。
# # x_pf = pf.transform(X) 对 X 进行多项式扩展，得到扩展后的特征集 x_pf。
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import PolynomialFeatures
# pf=PolynomialFeatures(degree=2)
# pf.fit(X)
# x_pf=pf.transform(X)
# x_train,x_test,y_train,y_test=train_test_split(x_pf,y,test_size=0.2,random_state=42)
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import numpy as np
#
# lr=LinearRegression()
# lr.fit(x_pf,y)
# # lr.score(x_test, y_test) 评估模型在测试集上的表现（返回测试集的 R² 分数）。
# # lr.score(x_train, y_train) 评估模型在训练集上的表现（返回训练集的 R² 分数）。
# print(f'R² score:',lr.score(x_pf,y))
# pred_y = lr.predict(x_pf)
# mse = mean_squared_error(y, pred_y)
# print(f"均方误差 (MSE): {mse}")
# rmse = np.sqrt(mse)
# print(f"均方根误差 (RMSE): {rmse}")
# mae = mean_absolute_error(y, pred_y)
# print(f"平均绝对误差 (MAE): {mae}")
#
# pred_y=lr.predict(x_pf)
# dic={"pred_y":pred_y,"org_y":y}
# predication = pd.DataFrame(dic)
#
# # 绘制预测值与真实值对比的折线图
# plt.figure(figsize=(10, 6))
# plt.plot(predication.index, predication["org_y"], label='真实值', marker='o')
# plt.plot(predication.index, predication["pred_y"], label='预测值', marker='x')
# plt.xlabel('Sample number')
# plt.ylabel('Quality Rating')
# plt.title('Predicted VS Origin values')
# plt.legend()
# plt.show()

#########################XGBoost/LASSO,用了train_test########################
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import numpy as np
#
# # 读取数据
# df = pd.read_csv("manufacturing.csv")
#
# # 数据预处理
# x = df.iloc[:, :-1]
# y = df["Quality_Rating"]
# df['index'] = df.index
#
# # 特征标准化
# ss = StandardScaler()
# ss.fit(x)
# X = pd.DataFrame(ss.transform(x), columns=x.columns)
#
# # 训练集和测试集分割
# # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# x_train, x_test, y_train, y_test, train_index, test_index = train_test_split(
#     X, y, df['index'], test_size=0.2, random_state=42
# )
#
# # # XGBoost 回归器
# # xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
# # xgb.fit(x_train, y_train)
# # y_pred = xgb.predict(x_test)
#
# # LASSO 回归
# from sklearn.linear_model import Lasso
# # 创建Lasso回归模型
# alpha = 0.1  # 正则化参数
# lasso = Lasso(alpha=alpha)
# lasso.fit(x_train, y_train)
# y_pred = lasso.predict(x_test)
#
#
#
# # 计算评估指标
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(f"均方误差 (MSE): {mse}")
# print(f"均方根误差 (RMSE): {rmse}")
# print(f"平均绝对误差 (MAE): {mae}")
# print(f"决定系数 (R²): {r2}")
#
# # 将预测值与真实值整理成DataFrame
# predication = pd.DataFrame({"index": test_index, "pred_y": y_pred, "org_y": y_test})
# predication.sort_values(by='index', inplace=True)
#
# # 绘制预测值与真实值对比的折线图
# plt.figure(figsize=(10, 6))
# plt.plot(predication.index, predication["org_y"], label='真实值', marker='o')
# plt.plot(predication.index, predication["pred_y"], label='预测值', marker='x')
# plt.xlabel('Sample number')
# plt.ylabel('Quality Rating')
# plt.title('Predicted VS Origin values')
# plt.legend()
# plt.show()


# ##################LSTM,考虑时序特性###########################
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
#
# # 读取数据
# df = pd.read_csv("manufacturing.csv")
#
# # 数据预处理
# x = df.iloc[:, :-1].values
# y = df["Quality_Rating"].values
#
# # 特征标准化
# ss = StandardScaler()
# x = ss.fit_transform(x)
#
#
# # 数据窗口化
# def create_dataset(X, y, time_steps=1):
#     Xs, ys = [], []
#     for i in range(len(X) - time_steps):
#         Xs.append(X[i:(i + time_steps)])
#         ys.append(y[i + time_steps])
#     return np.array(Xs), np.array(ys)
#
#
# time_steps = 25  # 你可以根据数据的时间序列特性调整这个值
# x_windowed, y_windowed = create_dataset(x, y, time_steps)
#
# # 转换为 PyTorch 张量
# x_tensor = torch.tensor(x_windowed, dtype=torch.float32)
# y_tensor = torch.tensor(y_windowed, dtype=torch.float32)
#
# # 训练集和测试集分割
# x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=42)
#
# # 创建 DataLoader
# train_dataset = TensorDataset(x_train, y_train)
# test_dataset = TensorDataset(x_test, y_test)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
#
# # 定义 LSTM 模型
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=1):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
#         c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out
#
#
# # 初始化模型、损失函数和优化器
# input_size = x_tensor.shape[2]
# hidden_size = 50
# output_size = 1
# num_layers = 1
#
# model = LSTMModel(input_size, hidden_size, output_size, num_layers)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 训练模型并记录训练和测试损失
# num_epochs = 100
# train_losses = []
# test_losses = []
#
# for epoch in range(num_epochs):
#     model.train()
#     train_loss_epoch = 0
#     for batch_x, batch_y in train_loader:
#         optimizer.zero_grad()
#         outputs = model(batch_x)
#         loss = criterion(outputs.squeeze(), batch_y)
#         loss.backward()
#         optimizer.step()
#         train_loss_epoch += loss.item() * batch_x.size(0)
#
#     train_loss_epoch /= len(train_loader.dataset)
#     train_losses.append(train_loss_epoch)
#
#     model.eval()
#     test_loss_epoch = 0
#     with torch.no_grad():
#         for batch_x, batch_y in test_loader:
#             outputs = model(batch_x)
#             loss = criterion(outputs.squeeze(), batch_y)
#             test_loss_epoch += loss.item() * batch_x.size(0)
#
#     test_loss_epoch /= len(test_loader.dataset)
#     test_losses.append(test_loss_epoch)
#
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss_epoch:.4f}, Test Loss: {test_loss_epoch:.4f}')
#
# # 绘制训练损失和测试损失曲线
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
# plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', marker='x')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Train and Test Loss Over Epochs')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # 测试模型
# model.eval()
# with torch.no_grad():
#     y_pred = []
#     y_true = []
#     for batch_x, batch_y in test_loader:
#         outputs = model(batch_x)
#         y_pred.append(outputs.squeeze().numpy())
#         y_true.append(batch_y.numpy())
#
#     y_pred = np.concatenate(y_pred)
#     y_true = np.concatenate(y_true)
#
# # 计算评估指标
# mse = mean_squared_error(y_true, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_true, y_pred)
# r2 = r2_score(y_true, y_pred)
#
# print(f"均方误差 (MSE): {mse}")
# print(f"均方根误差 (RMSE): {rmse}")
# print(f"平均绝对误差 (MAE): {mae}")
# print(f"决定系数 (R²): {r2}")
#
# # 将预测值与真实值整理成 DataFrame
# predication = pd.DataFrame({"pred_y": y_pred, "org_y": y_true})
#
# # 绘制预测值与真实值对比的折线图
# plt.figure(figsize=(10, 6))
# plt.plot(predication.index, predication["org_y"], label='真实值', marker='o')
# plt.plot(predication.index, predication["pred_y"], label='预测值', marker='x')
# plt.xlabel('Sample number')
# plt.ylabel('Quality Rating')
# plt.title('Predicted VS Origin values')
# plt.legend()
# plt.show()

# ##################三层MLP###########################
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
#
# # 读取数据
# df = pd.read_csv("combined_data.csv")
#
# # 数据预处理
# x = df.iloc[:, :-1].values
# y = df["Quality_Rating_Adjusted"].values
#
#
# # 特征标准化
# ss = StandardScaler()
# x = ss.fit_transform(x)
#
# # 转换为 PyTorch 张量
# x_tensor = torch.tensor(x, dtype=torch.float32)
# y_tensor = torch.tensor(y, dtype=torch.float32)
#
# # 训练集和测试集分割
# x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=42)
# train_dataset = TensorDataset(x_train, y_train)
# test_dataset = TensorDataset(x_test, y_test)
#
# # DataLoader with drop_last=True to handle the last incomplete batch
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)
#
# # 定义三层感知机模型
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(in_features=x.shape[1], out_features=128)  # 根据实际输入特征数量设置
#         self.fc2 = nn.Linear(in_features=128, out_features=64)
#         self.fc3 = nn.Linear(in_features=64, out_features=32)
#         self.fc4 = nn.Linear(in_features=32, out_features=1)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x.squeeze()
#
# model = MLP()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 训练模型
# def train(model, criterion, optimizer, train_loader, num_epochs=20):
#     model.train()
#     train_losses = []
#     for epoch in range(num_epochs):
#         epoch_loss = 0
#         for inputs, targets in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item() * inputs.size(0)
#         epoch_loss /= len(train_loader.dataset)
#         train_losses.append(epoch_loss)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
#     return train_losses
#
# train_losses = train(model, criterion, optimizer, train_loader)
#
# # 预测
# model.eval()
# y_pred = []
# y_true = []
# with torch.no_grad():
#     for inputs, targets in test_loader:
#         outputs = model(inputs)
#         y_pred.extend(outputs.cpu().numpy())
#         y_true.extend(targets.cpu().numpy())
#
# # 计算评估指标
# y_pred = np.array(y_pred)
# y_true = np.array(y_true)
# mse = mean_squared_error(y_true, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_true, y_pred)
# r2 = r2_score(y_true, y_pred)
#
# print(f"均方误差 (MSE): {mse}")
# print(f"均方根误差 (RMSE): {rmse}")
# print(f"平均绝对误差 (MAE): {mae}")
# print(f"决定系数 (R²): {r2}")
#
# # 将预测值与真实值整理成 DataFrame
# predication = pd.DataFrame({"pred_y": y_pred, "org_y": y_true})
#
# # 绘制预测值与真实值对比的折线图
# plt.figure(figsize=(10, 6))
# plt.plot(predication.index, predication["org_y"], label='真实值', marker='o')
# plt.plot(predication.index, predication["pred_y"], label='预测值', marker='x')
# plt.xlabel('Sample number')
# plt.ylabel('Quality Rating')
# plt.title('Predicted VS Origin values')
# plt.legend()
# plt.show()
#
# # 绘制训练损失
# plt.figure(figsize=(12, 6))
# plt.plot(train_losses, label='Train Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Train Loss')
# plt.legend()
# plt.show()

# #########多项式回归#########################
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 两两关系分布
# df=pd.read_csv("combined_data.csv")
# sns.pairplot(data=df)
# plt.show()
#
# # 分成x和y
# x=df.iloc[:,:-1]
# y=df["Quality_Rating_Adjusted"]
# # 每一列绘制箱型图
# for i, col in enumerate(x.columns):
#     plt.subplot(3, 3, i+1)
#     sns.boxplot(y=col, data=df)
#     plt.tight_layout()
# plt.show()
#
#
# from sklearn.preprocessing import StandardScaler
# ss=StandardScaler()
# ss.fit(x)
# X=pd.DataFrame(ss.transform(x),columns=x.columns)
# sns.heatmap(data=df.corr(),annot=True)
# # plt.xticks(rotation=45, ha='right')
# plt.show()
#
# # 初始化多项式特征生成器 pf，将特征数据扩展到二次项。
# # pf.fit(X) 根据标准化后的特征数据 X 学习数据的结构。
# # x_pf = pf.transform(X) 对 X 进行多项式扩展，得到扩展后的特征集 x_pf。
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import PolynomialFeatures
# pf=PolynomialFeatures(degree=2)
# pf.fit(X)
# x_pf=pf.transform(X)
# x_train,x_test,y_train,y_test=train_test_split(x_pf,y,test_size=0.2,random_state=42)
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import numpy as np
#
# lr=LinearRegression()
# lr.fit(x_pf,y)
# # lr.score(x_test, y_test) 评估模型在测试集上的表现（返回测试集的 R² 分数）。
# # lr.score(x_train, y_train) 评估模型在训练集上的表现（返回训练集的 R² 分数）。
# print(f'R² score:',lr.score(x_pf,y))
# pred_y = lr.predict(x_pf)
# mse = mean_squared_error(y, pred_y)
# print(f"均方误差 (MSE): {mse}")
# rmse = np.sqrt(mse)
# print(f"均方根误差 (RMSE): {rmse}")
# mae = mean_absolute_error(y, pred_y)
# print(f"平均绝对误差 (MAE): {mae}")
#
# pred_y=lr.predict(x_pf)
# dic={"pred_y":pred_y,"org_y":y}
# predication = pd.DataFrame(dic)
#
# # 绘制预测值与真实值对比的折线图
# plt.figure(figsize=(10, 6))
# plt.plot(predication.index, predication["org_y"], label='真实值', marker='o')
# plt.plot(predication.index, predication["pred_y"], label='预测值', marker='x')
# plt.xlabel('Sample number')
# plt.ylabel('Quality Rating')
# plt.title('Predicted VS Origin values')
# plt.legend()
# plt.show()



# ##################seq2seq#########################
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from tqdm import tqdm
# import torch
# from torch import nn
# import torch.utils.data as Data
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import warnings
# import torch.nn.functional as F
# from models import *
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 读取数据
# df = pd.read_csv("combined_data.csv")
#
# # 分成x和y
# data_x = df.iloc[:, :-1]
# data_y = df["Quality_Rating_Adjusted"]
#
# # 数据标准化
# scaler_x = StandardScaler()
# scaler_y = StandardScaler()
# data_x = scaler_x.fit_transform(data_x)
#
# # 对目标变量进行标准化
# data_y = scaler_y.fit_transform(data_y.values.reshape(-1, 1))
#
# data_4_x = []
# data_4_y = []
# time_step = 12
#
# for i in range(0, len(data_y) - time_step):
#     data_4_x.append(data_x[i:i + time_step])
#     data_4_y.append(data_y[i + time_step])
#
# data_4_x = np.array(data_4_x)
# data_4_y = np.array(data_4_y)
#
# x_train, x_test, y_train, y_test = train_test_split(data_4_x, data_4_y, test_size=0.2, random_state=42)
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
# Batch_Size = 256
# train_dataset = DataSet(x_train, y_train)
# test_dataset = DataSet(x_test, y_test)
#
# TrainDataLoader = Data.DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True, drop_last=True)
# TestDataLoader = Data.DataLoader(test_dataset, batch_size=Batch_Size, shuffle=True, drop_last=True)
#
# # 定义模型、损失函数和优化器
# model = LSTMDropout(input_size=9, hidden_size=128, num_layers=2, output_size=1).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.MSELoss().to(device)
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
# # 训练模型
# epochs = 100
# test_loss = []
# train_loss = []
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
# # 预测
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
#
# # 对预测结果进行反标准化处理
# y_pred = scaler_y.inverse_transform(np.array(y_pred).reshape(-1, 1)).squeeze()
# y_true = scaler_y.inverse_transform(np.array(y_true).reshape(-1, 1)).squeeze()
#
# # 画折线图显示
# plt.plot(y_true, label='True Values', marker='o')
# plt.plot(y_pred, label='Predicted Values', marker='x')
# plt.xlabel('Samples')
# plt.ylabel('Qualified Rate')
# plt.title('Qualified Rate based on CNNLSTMAttention')
# plt.legend()
# plt.show()
#
# # 计算评估指标
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#
# mse = mean_squared_error(y_true, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_true, y_pred)
# r2 = r2_score(y_true, y_pred)
#
# print(f"均方误差 (MSE): {mse}")
# print(f"均方根误差 (RMSE): {rmse}")
# print(f"平均绝对误差 (MAE): {mae}")
# print(f"决定系数 (R²): {r2}")


# ########################XGBoost/LASSO,用了train_test########################
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
#
#
# # 读取数据
# df = pd.read_csv("combined_data.csv")
#
# # 分成x和y
# x = df.iloc[:, :-1]
# y = df["Quality_Rating_Adjusted"]
# df['index'] = df.index
#
# # 特征标准化
# ss = StandardScaler()
# ss.fit(x)
# X = pd.DataFrame(ss.transform(x), columns=x.columns)
#
# # 训练集和测试集分割
# # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# x_train, x_test, y_train, y_test, train_index, test_index = train_test_split(
#     X, y, df['index'], test_size=0.2, random_state=42
# )
#
#
# # 最骄傲参数搜索
# # 定义网格搜索参数
# # 定义XGBoost回归模型
# xgb = XGBRegressor(objective='reg:squarederror')
#
# # 定义网格搜索参数
# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0]
# }
#
# # 使用GridSearchCV进行网格搜索
# grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(x_train, y_train)
#
# # 输出最佳参数
# best_params = grid_search.best_params_
# print(f"最佳的参数组合: {best_params}")
#
# # 使用最佳参数训练最终模型
# best_xgb = XGBRegressor(**best_params, objective='reg:squarederror')
# best_xgb.fit(x_train, y_train)
# y_pred = best_xgb.predict(x_test)


# # 定义随机森林回归模型
# rfr = RandomForestRegressor(n_estimators=100, random_state=42)
# rfr.fit(x_train, y_train)
# y_pred = rfr.predict(x_test)

# # XGBoost 回归器
# xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
# xgb.fit(x_train, y_train)
# y_pred = xgb.predict(x_test)

# # LASSO 回归
# from sklearn.linear_model import Lasso
# # 创建Lasso回归模型
# alpha = 0.1  # 正则化参数
# lasso = Lasso(alpha=alpha)
# lasso.fit(x_train, y_train)
# y_pred = lasso.predict(x_test)



# # 计算评估指标
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(f"均方误差 (MSE): {mse}")
# print(f"均方根误差 (RMSE): {rmse}")
# print(f"平均绝对误差 (MAE): {mae}")
# print(f"决定系数 (R²): {r2}")
#
# # 将预测值与真实值整理成DataFrame
# predication = pd.DataFrame({"index": test_index, "pred_y": y_pred, "org_y": y_test})
# predication.sort_values(by='index', inplace=True)
#
# # 绘制预测值与真实值对比的折线图
# plt.figure(figsize=(10, 6))
# plt.plot(predication.index, predication["org_y"], label='True', marker='o')
# plt.plot(predication.index, predication["pred_y"], label='Predicted', marker='x')
# plt.xlabel('Sample number')
# plt.ylabel('Quality Rating')
# plt.title('Predicted VS Origin values')
# plt.legend()
# plt.show()



#################新数据集seq2seq#################
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,\
                                mean_absolute_percentage_error
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm
import torch
from torch import nn
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from models import *
warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体（SimHei）
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
import seaborn as sns



# 定义Dataset和DataLoader
class DataSet(Data.Dataset):
    def __init__(self, data_inputs, data_targets):
        self.inputs = torch.FloatTensor(data_inputs)
        self.label = torch.FloatTensor(data_targets)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)


# 读取数据
# scaler = MinMaxScaler()
# df = pd.read_csv("manufacturing.csv")
# y_1 = df["Quality_Rating"][:3957].values.reshape(-1, 1)
# scaler.fit(y_1)
# y_1 = scaler.transform(y_1)
# df = pd.read_excel("quality.xlsx")
# y_2 = df["Qualified_rate"][:3957].values.reshape(-1, 1)
# scaler.fit(y_2)
# y_2 = scaler.transform(y_2)
# df = pd.read_csv("data_Y.csv")
# # y_3 = df["quality"][:3957]
# y_3 = df["quality"][:3957].values.reshape(-1, 1)
# scaler.fit(y_3)
# y_3 = scaler.transform(y_3)
# # y_array = np.array(y_1*y_2*y_3)
# # y_array = np.concatenate(tuple([y_1,y_2,y_3]),axis=1)
#
# df = pd.read_csv("manufacturing.csv")
# x1 = df.iloc[:3957, :-1]
# df = pd.read_excel("quality.xlsx")
# x2 = df.iloc[:3957, :-1]
# df = pd.read_csv("data_X.csv")
# x3 = df.iloc[:3957, 8:13]
#
# # Convert DataFrames to NumPy arrays and add a new dimension at axis 1
# x1_array = np.expand_dims(x1.to_numpy(), axis=1)
# x2_array = np.expand_dims(x2.to_numpy(), axis=1)
# x3_array = np.expand_dims(x3.to_numpy(), axis=1)

# x_array = []
# y_array = []
# time_step = 12
# for i in range(0, len(x) - time_step):
#     x_array.append(x[i:i + time_step])
#     y_array.append(y[i + time_step])
# x_array = np.array(x_array)
# y_array = np.array(y_array)



scaler = MinMaxScaler()
df = pd.read_csv("weather.csv")
y1 = df.iloc[:4000, -1].values.reshape(-1,1) #必须要二维才行
scaler.fit(y1)
y1 = scaler.transform(y1)
y2 = df.iloc[4000:8000, -1].values.reshape(-1,1) #必须要二维才行
scaler.fit(y2)
y2 = scaler.transform(y2)
y3 = df.iloc[8000:12000, -1].values.reshape(-1,1) #必须要二维才行
scaler.fit(y3)
y3 = scaler.transform(y3)


x1 = df.iloc[:4000, 1:-1]
scaler.fit(x1)
x1 = scaler.transform(x1)
x2 = df.iloc[4000:8000, 1:-1]
scaler.fit(x2)
x2 = scaler.transform(x2)
x3 = df.iloc[8000:12000, 1:-1]
scaler.fit(x3)
x3 = scaler.transform(x3)


# # 合并数据并创建标识列
# df_combined = pd.DataFrame(x1, columns=[col + '1' for col in df.columns[1:-1]])
# df_combined['target'] = 'x1'
# df_combined = df_combined.append(pd.DataFrame(x2, columns=[col + '2' for col in df.columns[1:-1]]),
#                                  ignore_index=True)
# df_combined['target'] = 'x2'
# df_combined = df_combined.append(pd.DataFrame(x3, columns=[col + '3' for col in df.columns[1:-1]]),
#                                  ignore_index=True)
# df_combined['target'] = 'x3'
#
# # 由于 y1, y2, y3 是目标变量，因此我们将它们与特征合并
# df_combined = df_combined.append(pd.DataFrame(y1, columns=['target_value1']), ignore_index=True)
# df_combined['target'] = 'y1'
# df_combined = df_combined.append(pd.DataFrame(y2, columns=['target_value2']), ignore_index=True)
# df_combined['target'] = 'y2'
# df_combined = df_combined.append(pd.DataFrame(y3, columns=['target_value3']), ignore_index=True)
# df_combined['target'] = 'y3'
#
# # 绘制 pairplot
# sns.pairplot(df_combined, hue='target')
# plt.show()


def preprocess_time(x1,x2,x3,y1,y2,y3):#和model有关,3个时间步
    x1 = np.expand_dims(x1, axis=1)
    x2 = np.expand_dims(x2, axis=1)
    x3 = np.expand_dims(x3, axis=1)
    x_array = np.concatenate([x1, x2, x3], axis=1)
    y_array = np.concatenate(([y1, y2, y3]), axis=1)
    return x_array,y_array

def preprocess_regression(x1, x2, x3, y1, y2, y3):#全部平铺
    x_array = np.concatenate([x1, x2, x3], axis=1)
    y_array = np.concatenate(([y1, y2, y3]), axis=1)
    return x_array, y_array

def model_train(x1,x2,x3,y1,y2,y3):
    x_array, y_array = preprocess_time(x1,x2,x3,y1,y2,y3)
    x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.2, random_state=42)
    Batch_Size = 128
    train_dataset = DataSet(x_train, y_train)
    test_dataset = DataSet(x_test, y_test)

    TrainDataLoader = Data.DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True, drop_last=True)
    TestDataLoader = Data.DataLoader(test_dataset, batch_size=Batch_Size, shuffle=True, drop_last=True)

    # 定义模型、损失函数和优化器
    model = Seq2SeqAttention_dropout(input_size=x_array.shape[-1], hidden_size=128, num_layers=2, output_size=y_array.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss().to(device)

    def test(loader):
        model.eval()
        with torch.no_grad():
            val_epoch_loss = []
            for inputs, targets in loader:
                inputs, targets = inputs.to(device).float(), targets.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_epoch_loss.append(loss.item())
            return np.mean(val_epoch_loss)

    # 训练模型
    epochs = 100
    test_loss = []
    train_loss = []
    best_test_loss = float('inf')

    for epoch in tqdm(range(epochs)):
        model.train()
        train_epoch_loss = []
        for inputs, targets in TrainDataLoader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
        train_loss.append(np.mean(train_epoch_loss))
        test_epoch_loss = test(TestDataLoader)
        test_loss.append(test_epoch_loss)
        print(f"Epoch: {epoch + 1}, Train Loss: {np.mean(train_epoch_loss)}, Test Loss: {test_epoch_loss}")

    fig = plt.figure(figsize=(10, 7))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(range(len(train_loss)), train_loss, label='Train Loss')
    plt.plot(range(len(test_loss)), test_loss, label='Test Loss')
    plt.title('Loss based on Seq2SeqAttention_dropout')
    plt.legend()
    plt.show()

    # 加载最佳模型并进行预测
    # model.load_state_dict(torch.load('best_LSTM_model.pth'))
    model.eval()

    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, targets in TestDataLoader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            outputs = model(inputs)
            y_pred.extend(outputs.squeeze().cpu().numpy())
            y_true.extend(targets.cpu().numpy())
    return y_true,y_pred


def model_regression(x1, x2, x3, y1, y2, y3):
    x_array, y_array = preprocess_regression(x1, x2, x3, y1, y2, y3)
    x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.2, random_state=42)
    regress = Lasso(alpha=0.1)
    # regress = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
    regress.fit(x_train, y_train)
    y_pred = regress.predict(x_test)
    y_true = y_test
    return y_true, y_pred

def model_polymental_regression(x1, x2, x3, y1, y2, y3):
    x_array, y_array = preprocess_regression(x1, x2, x3, y1, y2, y3)
    pf = PolynomialFeatures(degree=2)
    pf.fit(x_array)
    x_pf = pf.transform(x_array)
    x_train, x_test, y_train, y_test = train_test_split(x_pf, y_array, test_size=0.2, random_state=42)
    regress = LinearRegression()
    regress.fit(x_train, y_train)
    y_pred = regress.predict(x_test)
    y_true = y_test
    return y_true, y_pred



# y_test,y_pred = model_train(x1,x2,x3,y1,y2,y3)
y_true, y_pred = model_polymental_regression(x1, x2, x3, y1, y2, y3)
y_true = np.array(y_true)
y_pred = np.array(y_pred)
# 画折线图显示
for i in range(y_true.shape[-1]):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(y_true[:,i].squeeze(), label='True Values', marker='o')
    plt.plot(y_pred[:,i].squeeze(), label='Predicted Values', marker='x')
    plt.xlabel('Samples')
    plt.ylabel('Qualified Rate')
    plt.title('Qualified Rate based on Seq2SeqAttention_dropout in stage {i}'.format(i=i+1))
    plt.legend()
    plt.show()

# 计算评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

for i in range(y_true.shape[-1]):
    mse = mean_squared_error(y_true[:,i], y_pred[:,i])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true[:,i], y_pred[:,i])
    mape = mean_absolute_percentage_error(y_true[:,i], y_pred[:,i])
    r2 = r2_score(y_true[:,i], y_pred[:,i])

    print(f"均方误差 (MSE) 阶段: {mse},阶段{i}")
    print(f"均方根误差 (RMSE): {rmse}，阶段{i}")
    print(f"平均绝对误差 (MAE): {mae}，阶段{i}")
    print(f"百分比误差(MAPE):{mape}，阶段{i}")
    print(f"决定系数 (R²): {r2}，阶段{i}")


