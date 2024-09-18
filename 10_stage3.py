import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 两两关系分布
df=pd.read_csv("manufacturing.csv")
sns.pairplot(data=df)
plt.show()

# 分成x和y
x=df.iloc[:,:-1]
y=df["Quality_Rating"]
# 每一列绘制箱型图
for i, col in enumerate(x.columns):
    plt.subplot(2, 3, i+1)
    sns.boxplot(y=col, data=df)
    plt.tight_layout()
plt.show()


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(x)
X=pd.DataFrame(ss.transform(x),columns=x.columns)
sns.heatmap(data=df.corr(),annot=True)
# plt.xticks(rotation=45, ha='right')
plt.show()

# 初始化多项式特征生成器 pf，将特征数据扩展到二次项。
# pf.fit(X) 根据标准化后的特征数据 X 学习数据的结构。
# x_pf = pf.transform(X) 对 X 进行多项式扩展，得到扩展后的特征集 x_pf。
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
pf=PolynomialFeatures(degree=2)
pf.fit(X)
x_pf=pf.transform(X)
x_train,x_test,y_train,y_test=train_test_split(x_pf,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
# lr.score(x_test, y_test) 评估模型在测试集上的表现（返回测试集的 R² 分数）。
# lr.score(x_train, y_train) 评估模型在训练集上的表现（返回训练集的 R² 分数）。
print(lr.score(x_test,y_test),lr.score(x_train,y_train))
pred_y=lr.predict(x_pf)
dic={"pred_y":pred_y,"org_y":y}
predication=pd.DataFrame(dic)

# 绘制预测值与真实值对比的折线图
plt.figure(figsize=(10, 6))
plt.plot(predication.index, predication["org_y"], label='真实值', marker='o')
plt.plot(predication.index, predication["pred_y"], label='预测值', marker='x')
plt.xlabel('Sample number')
plt.ylabel('Quality Rating')
plt.title('Predicted VS Origin values')
plt.legend()
plt.show()