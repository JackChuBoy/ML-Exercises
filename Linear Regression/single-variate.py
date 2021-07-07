import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 設定x, y資料
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

print("X:{}".format(x))
print("Y:{}".format(y))

#建立回歸模型並運算
regx = linear_model.LinearRegression()
regx.fit(x, y)

#取得預測值
y_pred = regx.predict(x)

#相關係數
print('Coefficients: \n', regx.coef_)

#均方誤差
print('Mean squared error: %.2f'
      % mean_squared_error(y, y_pred))

#決定係數
print('Coefficient of determination: %.2f'
      % r2_score(y, y_pred))

#繪製圖型
plt.scatter(x, y,  color='black')
plt.plot(x, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()