import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import torch.nn.functional as F
from models import *

warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据
data = pd.read_excel("quality.xlsx")
# data = pd.read_csv("manufacturing.csv")
data = data.fillna(0)
data_x = data[["Temperature", "Humidity", "Proficiency", "Usage_Time", "fixture_accuracy"]].values
data_y = data['Qualified_rate'].values

# 数据分组
data_4_x = []
data_4_y = []
time_step = 12
for i in range(0, len(data_y) - time_step):
    data_4_x.append(data_x[i:i + time_step])
    data_4_y.append(data_y[i + time_step])
data_4_x = np.array(data_4_x)
data_4_y = np.array(data_4_y)

x_train, x_test, y_train, y_test = train_test_split(data_4_x, data_4_y, test_size=0.2, random_state=42)


# 定义Dataset和DataLoader
class DataSet(Data.Dataset):
    def __init__(self, data_inputs, data_targets):
        self.inputs = torch.FloatTensor(data_inputs)
        self.label = torch.FloatTensor(data_targets)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)


Batch_Size = 128
train_dataset = DataSet(x_train, y_train)
test_dataset = DataSet(x_test, y_test)

TrainDataLoader = Data.DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True, drop_last=True)
TestDataLoader = Data.DataLoader(test_dataset, batch_size=Batch_Size, shuffle=True, drop_last=True)

# 定义模型、损失函数和优化器
model = LSTM(input_size=5, hidden_size=128, num_layers=2, output_size=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss().to(device)


def test(loader):
    model.eval()
    with torch.no_grad():
        val_epoch_loss = []
        for inputs, targets in loader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float().view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_epoch_loss.append(loss.item())
        return np.mean(val_epoch_loss)


# 训练模型
epochs = 30
test_loss = []
train_loss = []
best_test_loss = float('inf')

for epoch in tqdm(range(epochs)):
    model.train()
    train_epoch_loss = []
    for inputs, targets in TrainDataLoader:
        inputs, targets = inputs.to(device).float(), targets.to(device).float().view(-1, 1)
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

    if test_epoch_loss < best_test_loss:
        best_test_loss = test_epoch_loss
        best_model = model.state_dict()
        torch.save(best_model, 'best_LSTM_model.pth')

# 画损失图
fig = plt.figure(facecolor='white', figsize=(10, 7))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(range(len(train_loss)), train_loss, label='Train Loss')
plt.plot(range(len(test_loss)), test_loss, label='Test Loss')
plt.title('Loss based on CNNLSTMAttention')
plt.legend()
plt.show()

# 加载最佳模型并进行预测
model.load_state_dict(torch.load('best_LSTM_model.pth'))
model.eval()

y_pred = []
y_true = []
with torch.no_grad():
    for inputs, targets in TestDataLoader:
        inputs, targets = inputs.to(device).float(), targets.to(device).float()
        outputs = model(inputs)
        y_pred.extend(outputs.squeeze().cpu().numpy())
        y_true.extend(targets.cpu().numpy())

# 画折线图显示
plt.plot(y_true, label='True Values', marker='o')
plt.plot(y_pred, label='Predicted Values', marker='x')
plt.xlabel('Samples')
plt.ylabel('Qualified Rate')
plt.title('Qualified Rate based on CNNLSTMAttention')
plt.legend()
plt.show()

# 计算评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"均方误差 (MSE): {mse}")
print(f"均方根误差 (RMSE): {rmse}")
print(f"平均绝对误差 (MAE): {mae}")
print(f"决定系数 (R²): {r2}")

# ##################多项式回归,没用train_test###########################
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import seaborn as sns
#
#
# # 读取数据
# df = pd.read_excel("quality.xlsx")
# df = df.fillna(0)
# x = df[["Temperature","Humidity","Proficiency","Usage_Time","fixture_accuracy"]]
# y = df['Qualified_rate']
#
#
# # 两两关系分布
# sns.pairplot(data=df)
# plt.show()
#
#
# # 每一列绘制箱型图
# for i, col in enumerate(df.columns):
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
# plt.plot(predication.index, predication["org_y"], label='True', marker='o')
# plt.plot(predication.index, predication["pred_y"], label='Predicted', marker='x')
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
# df = pd.read_excel("quality.xlsx")
# df = df.fillna(0)
# x = df[["Temperature","Humidity","Proficiency","Usage_Time","fixture_accuracy"]]
# y = df['Qualified_rate']
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
