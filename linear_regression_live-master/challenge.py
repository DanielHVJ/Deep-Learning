#%%
import pandas as pd
import numpy as np
from sklearn import linear_model as model
import matplotlib.pyplot as plt

#%%
dt = pd.read_csv('datacha.csv', header=None, names=['X', 'y'])
dt.head()

# %%
X = pd.DataFrame(dt.iloc[:, 0])
y = pd.DataFrame(dt.iloc[:, 1])

# %%
regression = model.LinearRegression()
regression.fit(X, y)

#%%
print('coefficients: ', regression.coef_)
print('Mean squared error: %.2f ' % np.mean((regression.predict(X) - y)**2))
print('Variance score: %.2f' % regression.score(X,y))
# %%
plt.scatter(X, y, color='orange')
plt.plot(X, regression.predict(X))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Challenge Dataset')
plt.show()
# %%
## BONUS

%matplotlib inline

import matplotlib.pyplot as plt
from sklearn import linear_model as model
from sklearn import datasets
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib as mpl
# %%
iris = datasets.load_iris()

# %%
X = iris.data[:,1:3]
y = iris.data[:, 0]

# %%
linear = model.LinearRegression()
linear.fit(X,y)

# %%
print('coefficients: ', linear.coef_)
print('Mean squared error: %.2f ' % np.mean((linear.predict(X)-y)**2))
print('Variance score: %.2f' % linear.score(X,y))
# %%
fig = plt.figure()
fig.set_size_inches(9,7)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1], y, c='g', marker= 'o')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Species')
ax.set_title('Orignal Dataset')
ax.view_init(10, -45)

fig1 = plt.figure()
fig1.set_size_inches(9,7)
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], linear.predict(X), c='b', marker= '*')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Species')
ax.set_title('Predicted Dataset')
ax.view_init(10, -45)
# %%
