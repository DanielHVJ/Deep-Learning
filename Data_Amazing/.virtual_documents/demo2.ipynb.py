#Here be data

import numpy as np
x = np.array([[-1, -1, -1,5,3,3], [-2, -1, -1.5,4,53,3], [-3, -2, -2,3,4,2]])
x


## Visualization in Original Dimensional Space

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode()

trace1 = Scatter3d(x = x[0], y = x[1], z = x[2],  mode = 'markers')
iplot([trace1], link_text = 'Plot')


#Normalize the data

x_scaled = np.array(list(map(lambda y: (y - np.mean(y)) /np.std(y), x.T))).T
x_scaled


#Get the covariance matrix and calculate its eigenvalues and eigenvectors.

cov_matrix = np.cov(x_scaled.T)
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
print(cov_matrix)


#Get the top k eigenvectors of the covariance matrix where k is the size of the desired subspace.

eigen_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eigen_pairs.sort(reverse = True)

for (eig_val, eig_vec) in eigen_pairs:
    print('Eigen value: {0}\nEigen vector: {1}'.format(eig_val, eig_vec), end='\n\n')
    
projection_matrix = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))

print('Projection Matrix:\n\n', projection_matrix)


#Project the data into this lower subspace.

x_pca_manual = np.dot(x_scaled, projection_matrix)
x_pca_manual


#Explained Variance

eig_vals.sort()

pca_eig_vals = np.round(eig_vals[::-1][:-1]/sum(eig_vals),3)

print('Explained Variance By Principal Component: ',pca_eig_vals)
print('Total Explained Variance: ',sum(pca_eig_vals))


# Visualization in New Subspace

scatter = Scatter(x = x_pca_manual[:, 0], y = x_pca_manual[:, 1], mode = 'markers')
layout = Layout(xaxis=dict(title='1st P.C'), yaxis=dict(title='2nd P.C'))
iplot(Figure(data=[scatter], layout=layout))


#PCA In Sklearn

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

x_scaled = StandardScaler().fit_transform(x)
pca = PCA(n_components = 2)

x_pca = pca.fit_transform(x_scaled)

print(x_pca, end = '\n\n')
print(pca.explained_variance_ratio_)


#Inpect The Data

import pandas as pd         

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', 
                      header = None)
cols = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity',  'Magnesium', 'Total phenols', 
        'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 
        'OD280/OD315 of diluted wines', 'Proline']
df_wine.columns = cols

print('Shape: ',df_wine.shape, end ='\n\n')
df_wine.head(5)


#Baseline accuracy

df_wine.Class.value_counts(normalize = True)


#Correlation Heat Map
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic("matplotlib", " inline")

sns.heatmap(df_wine.corr())
plt.show()


# P.C.A Logistic Regression with Two Components

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

pca_2_comps = PCA(n_components = 2)
X_train_pca_2_comps = pca_2_comps.fit_transform(X_train_std)

print('Variance Explained Per Principal Component:', pca_2_comps.explained_variance_ratio_, end = '\n\n')
print('Total Variance Explained:', np.sum(pca_2_comps.explained_variance_ratio_), end = '\n\n')

clf_logreg = LogisticRegression(max_iter = 400)
clf_logreg.fit(X_train_pca_2_comps, y_train)
pca_2_comps_score = clf_logreg.score(X_train_pca_2_comps, y_train)
print('Training Accuracy: ', pca_2_comps_score)


import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " inline")
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_train_pca_2_comps, y_train, clf = clf_logreg)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Decision Boundaries')
plt.legend(loc = 'upper left')
plt.show()


plt.scatter(pca_2_comps.components_.T[:,0], pca_2_comps.components_.T[:,1])
plt.title('Feature Contribution to Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')

for i, xy in enumerate(pca_2_comps.components_.T):                                  
    plt.annotate('(get_ipython().run_line_magic("s)'", " % cols[i+1], xy=xy, textcoords='data')")
plt.show()


# Comparision Vs Accuracy of Pairs of Features

from itertools import combinations
accuracy = []

for i in combinations([i for i in range(0, 13)], r = 2):
    clf_logreg.fit(X_train[:, i], y_train)
    if clf_logreg.score(X_train[:, i], y_train) >= pca_2_comps_score:
        accuracy.append((i, clf_logreg.score(X_train, y_train)))
        
print('Number of Two Feature Combinations Which Outperform 2 Comp. P.C.A: ', len(accuracy))


#Naive Model Accuracy

clf_logreg.fit(X_train, y_train)
clf_logreg.score(X_train, y_train)


#Explained Variance Ratio and Scree Test

pca_full = PCA(n_components = None)
pca_full.fit(X_train_std)

plt.bar(range(1,14), 
        pca_full.explained_variance_ratio_, 
        alpha = 0.5, 
        align = 'center', 
        label = 'individual explained variance', color='green')
plt.step(range(1,14), 
         np.cumsum(pca_full.explained_variance_ratio_), 
         where = 'mid', 
         label = 'cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc = 'best')
plt.show()


# Kaiser-Harris Criterion

def kaiser_harris_criterion(cov_mat):
    e_vals, _ = np.linalg.eig(cov_mat)
    return len(e_vals[e_vals > 1])

print('Kaiser-Harris Criterion: Use {} principal components.'.format(kaiser_harris_criterion(np.cov(X_train_std.T))))


#Smart P.C.A Logistic Regression vs Naive Logistic Regression

pca_4_comps = PCA(n_components = 4)
X_train_pca_4_comps = pca_4_comps.fit_transform(X_train_std)

clf_logreg.fit(X_train_pca_4_comps, y_train)
print('P.C.A Logistic Regression Training Accuracy:',clf_logreg.score(X_train_pca_4_comps, y_train))
print('P.C.A Logistic Regression Testing Accuracy:',clf_logreg.score(pca_4_comps.transform(X_test_std), y_test))
print('\n')

clf_logreg.fit(X_train, y_train)
print('Naive Logistic Regression Testing Accuracy:',clf_logreg.score(X_train, y_train))
print('Naive Logistic Regression Testing Accuracy:',clf_logreg.score(X_test, y_test))


#Get The Data

#Note: This code is different from the talk. There I was reading in the data locally. Here I'm doing it from the url.

import requests, zipfile, io
res_kdd = requests.get('http://kdd.org/cupfiles/KDDCupData/1999/kddcup.data_10_percent.zip')
file_kdd = zipfile.ZipFile(io.BytesIO(res_kdd.content))
file_kdd_access = file_kdd.open("kddcup.data_10_percent.txt")

cols = ["duration","protocol_type","service","flag","src_bytes", "dst_bytes","land",
        "wrong_fragment","urgent","hot","num_failed_logins", "logged_in","num_compromised",
        "root_shell","su_attempted","num_root", "num_file_creations","num_shells",
        "num_access_files","num_outbound_cmds", "is_host_login","is_guest_login","count",
        "srv_count","serror_rate", "srv_serror_rate","rerror_rate","srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate","srv_diff_host_rate","dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate","dst_host_srv_rerror_rate","TARGET"]

kdd_data = pd.read_csv(file_kdd_access, header=None, names=cols, low_memory=False)
kdd_data.head()


print('10% of the data is {} data points and has {} columns.'.format(len(kdd_data), len(kdd_data.columns)), end = '\n\n')
print('Missingness: ', end = '\n\n')
print(kdd_data.isnull().sum())


print('There are {} unique targets'.format(len(kdd_data.TARGET.unique())), end = '\n\n')
kdd_data.TARGET.value_counts(normalize=True)


#Adjusted Baseline Accuracy

kdd_data['BINARY_TARGET'] = kdd_data['TARGET'].map(lambda x: x if x=='normal.' else 'abnormal.')
kdd_data.BINARY_TARGET.value_counts(normalize=True)


#Column Types

kdd_data.dtypes


#Discrepancies

weird_cols = ['num_root', 'num_file_creations', 'num_shells']

for col in weird_cols:
    print(col)
    print(list(filter(lambda x: x[1]get_ipython().run_line_magic("2==0,", " [(x,x.isdigit()) for x in kdd_data[col].values])))")
    print('\n')


#Fix Discrepancies: Replace By Most Frequent Value

#Side Note: As people mention in the talk, dropping these rows is an option to consider as the whole row could be contaminated
#if this is a data reading error.

kdd_data.loc[kdd_data.num_root == 'tcp', 'num_root'] = kdd_data.num_root.value_counts().index[0]
kdd_data.loc[kdd_data.num_file_creations == 'http', 'num_file_creations'] = kdd_data.num_file_creations.value_counts().index[0]
kdd_data.loc[kdd_data.num_shells == 'SF', 'num_shells'] = kdd_data.num_shells.value_counts().index[0]

kdd_data.loc[:, weird_cols] = kdd_data.loc[:, weird_cols].apply(pd.to_numeric)


#su_attempted: Unique values are 0,1,2 but should be 0,1

kdd_data.loc[kdd_data.su_attempted == 2, 'su_attempted'] = 0


#Numerical Feature Summary

kdd_data_num_features = kdd_data.iloc[:,:-2].select_dtypes(exclude = ['object'])
kdd_data_cat_features = kdd_data.iloc[:,:-2].select_dtypes(include = ['object'])

kdd_data_num_features.describe()


#Categorical Feature Summary

kdd_data_cat_features.describe()


#correlation Heat Map

sns.heatmap(kdd_data_num_features.corr())
plt.show()


# Encoding Categorical Features

def column_encoding(df_num, df_cat):
    for i in range(len(df_cat.columns)):
        df_num = pd.concat([df_num, pd.get_dummies(df_cat.iloc[:,i])], axis=1)
    return df_num

kdd_features_dummied = column_encoding(kdd_data_num_features, kdd_data_cat_features)
len(kdd_features_dummied.columns)

kdd_features_dummied.head()


import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " inline")
 
scaled_X_dummied = StandardScaler().fit_transform(kdd_features_dummied)
pca_kdd = PCA(n_components = 2)
scaled_X_dummied_pca_2 = pca_kdd.fit_transform(scaled_X_dummied)

normal_indices = kdd_data[kdd_data.TARGETget_ipython().getoutput("='normal.'].index.values")
abnormal_indices = kdd_data[kdd_data.TARGET=='normal.'].index.values
plt.scatter(scaled_X_dummied_pca_2[normal_indices,0], scaled_X_dummied_pca_2[normal_indices,1], c= 'b', )
plt.scatter(scaled_X_dummied_pca_2[abnormal_indices,0], scaled_X_dummied_pca_2[abnormal_indices,1], c= 'r')
plt.title('Network Intrusion In 2D')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(['normal', 'abnormal'])
plt.show()


#Choose Subspace Dimension
k_harris_rec = kaiser_harris_criterion(np.cov(scaled_X_dummied.T))
print('Kaiser-Harris Criterion: Use {} principal components.'.format(k_harris_rec))

pca_kdd.set_params(n_components = k_harris_rec)
scaled_X_dummied_pca_k_harris = pca_kdd.fit_transform(scaled_X_dummied)


kdd_data.head()

y_train = np.where(kdd_data['BINARY_TARGET']=='normal.', 1, 0)

y_train.shape

y_train2 = y_train[0:30000]

scaled2_X_dummied_pca_k_harris = scaled_X_dummied_pca_k_harris[0:30000]


#Testing A Classifier

import sklearn.model_selection as cv
from sklearn.model_selection import GridSearchCV

# y_train = kdd_data.iloc[:,-1].map(lambda x: 1 if x=='normal.' else 0)
stratified_divide = cv.StratifiedKFold(10)
clf_log_reg2 = LogisticRegression(max_iter=50, n_jobs=4)
clf_log_reg2_cv_score = np.mean(cv.cross_val_score(clf_log_reg2, 
                                                   scaled2_X_dummied_pca_k_harris, 
                                                   y_train2, 
                                                   cv = stratified_divide, 
                                                   scoring = 'accuracy'))
print(clf_log_reg2_cv_score)



#Confusion Matrix

import sklearn.metrics as met
cm = met.confusion_matrix(y_train2, 
                     (clf_log_reg2.fit(scaled2_X_dummied_pca_k_harris, y_train2)
                                  .predict(scaled2_X_dummied_pca_k_harris)))

acc =met.accuracy_score(y_train2,(clf_log_reg2.fit(scaled2_X_dummied_pca_k_harris, y_train2)
                                  .predict(scaled2_X_dummied_pca_k_harris)))

print(cm, end='\n\n')
print(acc)
