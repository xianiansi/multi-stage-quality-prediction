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

# ########################XGBoost/LASSO,用了train_test########################
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
# # df = pd.read_excel("grouped_output_with_operation_time_v3.xlsx")
#
# # 数据预处理
# x = df.iloc[:, :-1]
# # y = df["Qualified_rate"]
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
# # XGBoost 回归器
# xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
# xgb.fit(x_train, y_train)
# y_pred = xgb.predict(x_test)
#
# # # LASSO 回归
# # from sklearn.linear_model import Lasso
# # # 创建Lasso回归模型
# # alpha = 0.1  # 正则化参数
# # lasso = Lasso(alpha=alpha)
# # lasso.fit(x_train, y_train)
# # y_pred = lasso.predict(x_test)
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


##################model,考虑时序特性###########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models import *



# 读取数据
df = pd.read_csv("manufacturing.csv")
# df = pd.read_excel("grouped_output_with_operation_time_v3.xlsx")


# 数据预处理
x = df.iloc[:, :-1].values
y = df["Quality_Rating"].values
# y = df["Qualified_rate"].values


# 特征标准化
ss = StandardScaler()
x = ss.fit_transform(x)


# 数据窗口化
# def create_dataset(X, y, time_steps=1):
#     Xs, ys = [], []
#     for i in range(len(X) - time_steps):
#         Xs.append(X[i:(i + time_steps)])
#         ys.append(y[i + time_steps])
#     return np.array(Xs), np.array(ys)

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps+1)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


time_steps = 25  # 你可以根据数据的时间序列特性调整这个值
x_windowed, y_windowed = create_dataset(x, y, time_steps)

# 转换为 PyTorch 张量
x_tensor = torch.tensor(x_windowed, dtype=torch.float32)
y_tensor = torch.tensor(y_windowed, dtype=torch.float32)

# 训练集和测试集分割
x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=42)

# 创建 DataLoader
batch_size = 128
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# 初始化模型、损失函数和优化器
input_size = x_tensor.shape[2]
hidden_size = 128
output_size = 1
num_layers = 1

model = Seq2SeqModel(input_size, hidden_size, num_layers,output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 训练模型并记录训练和测试损失
num_epochs = 100
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss_epoch = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item() * batch_x.size(0)

    train_loss_epoch /= len(train_loader.dataset)
    train_losses.append(train_loss_epoch)

    model.eval()
    test_loss_epoch = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            test_loss_epoch += loss.item() * batch_x.size(0)

    test_loss_epoch /= len(test_loader.dataset)
    test_losses.append(test_loss_epoch)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss_epoch:.4f}, Test Loss: {test_loss_epoch:.4f}')

# 绘制训练损失和测试损失曲线
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# 测试模型
model.eval()
with torch.no_grad():
    y_pred = []
    y_true = []
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        y_pred.append(outputs.squeeze().numpy())
        y_true.append(batch_y.numpy())

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

# 计算评估指标
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"均方误差 (MSE): {mse}")
print(f"均方根误差 (RMSE): {rmse}")
print(f"平均绝对误差 (MAE): {mae}")
print(f"决定系数 (R²): {r2}")

# 将预测值与真实值整理成 DataFrame
predication = pd.DataFrame({"pred_y": y_pred, "org_y": y_true})

# 绘制预测值与真实值对比的折线图
plt.figure(figsize=(10, 6))
plt.plot(predication.index, predication["org_y"], label='True', marker='o')
plt.plot(predication.index, predication["pred_y"], label='Predicted', marker='x')
plt.xlabel('Sample number')
plt.ylabel('Quality Rating')
plt.title('Predicted VS Origin values')
plt.legend()
plt.show()


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
#         self.fc1 = nn.Linear(in_features=5, out_features=128)  # 输入特征数量
#         self.fc2 = nn.Linear(in_features=128, out_features=64)
#         self.fc3 = nn.Linear(in_features=64, out_features=32)
#         self.fc4 = nn.Linear(in_features=32, out_features=1)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x
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
#             loss = criterion(outputs.view(-1), targets)
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
#         y_pred.extend(outputs.view(-1).cpu().numpy())
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
