import os
import pandas as pd
import numpy as np
# np.set_printoptions(precision=5)

# 读取数据 Load data
data_path = os.path.join(os.getcwd(), "machine-learning\\data\\titanic-train.csv")
dataset = pd.read_csv(data_path, skipinitialspace=True)

dataset.columns.values
dataset.head(3)

###########################
# 对数据进行一些preprocessing：丢弃一些不用的字段

# We need to drop some insignificant features and map the others.
# Ticket number and fare should not contribute much to the performance of our models.
# Name feature has titles (e.g., Mr., Miss, Doctor) included.
# Gender is definitely important.
# Port of embarkation may contribute some value.
# Using port of embarkation may sound counter-intuitive; however, there may 
# be a higher survival rate for passengers who boarded in the same port.

dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
dataset = dataset.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1)

# 计算sex和title的频率表-compute a cross-tabulation of sex and title: how many female and male for each title
pd.crosstab(dataset['Title'], dataset['Sex'])

#############################
# 对数据进行一些preprocessing：整理title字段
# We will replace many titles with a more common name, English equivalent,
# or reclassification
dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# 按title分组统计Survived的平均值
# 结果显示 Mrs > Miss > Master的生存几率
dataset[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

#################################
# # 对数据进行一些preprocessing：下面对categorical variable进行code。
# 注意命令使用方法: dataframe['field-name'].map(map-variable).astype(value-type)
# Now we will map alphanumerical categories to numbers
title_mapping = { 'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 5 }
gender_mapping = { 'female': 1, 'male': 0 }
port_mapping = { 'S': 0, 'C': 1, 'Q': 2 }

# Map title
dataset['Title'] = dataset['Title'].map(title_mapping).astype(int)

# Map gender
dataset['Sex'] = dataset['Sex'].map(gender_mapping).astype(int)

# Map port
freq_port = dataset.Embarked.dropna().mode()[0]
dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
dataset['Embarked'] = dataset['Embarked'].map(port_mapping).astype(int)

# Fix missing age values
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].dropna().median())

dataset.head()

######################################
# 使用四种方法进行survival rate的预测，并使用k-fold评估
# -Logistic regression with varying numbers of polynomials
# -Support vector machine with a linear kernel
# -Support vector machine with a polynomial kernel
# -Neural network
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 从数据集中建立X和Y-Prepare the data
# array X of size (n_samples, n_features), which holds the training samples represented as floating point feature vectors; 
# array y of size (n_samples,), which holds the target values (class labels) for the training samples
X = dataset.drop(['Survived'], axis = 1).values
y = dataset[['Survived']].values

# 对X进行z-score标准化
# StandardScaler.fit_transform(): transform your data such that its distribution will have a mean value 0 and standard deviation of 1. 
# Given the distribution of the data, each value in the dataset will have the sample mean value subtracted, and 
# then divided by the standard deviation of the whole dataset.
# Fit(): Method calculates the parameters μ and σ and saves them as internal objects.
# Transform(): Method using these calculated parameters apply the transformation to a particular dataset.
# Fit_transform(): joins the fit() and transform() method for transformation of dataset.
X = StandardScaler().fit_transform(X)

# test_size代表的是test dataset占整个数据集的比例
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = None)

# Prepare cross-validation (cv)
cv = KFold(n_splits = 5, random_state = None)

# ?
# Performance

# a multi-layer perceptron (MLP) algorithm
# 因为neural network面对的问题很多情况下都是non-convex的，所以极有可能存在多个局部最优解，因此随机初始化权重的时候，不同的权重会导致训练处不同的网络，因此也有不同的精度
# 我们常说的调参数指的是调整超参数(hyperparameters, 关于模型的参数),包括学习速率，迭代次数，层数，每层神经元的个数
# 使用MLP前建议先标准化(StandardScaler().fit_transform(X))，因为MLP is sensitive to feature scaling.
classifier = MLPClassifier(\
                # para: hidden_layer_sizes=(100,) if you want only 1 hidden layer with 100 hidden units.
                hidden_layer_sizes=(100,),\
                # para: alpha是用来设置正则化以限制overfitting. Alpha defines L2 penalty (regularization term) parameter.
                alpha=0.01,\
                # max_iter和tol设置训练的收敛条件。max_iter定义iteration次数，tol定义
                max_iter=200,\
                tol=0.001,\
                # para: random_state=1将随机数的seed设为1，这样每次生成的随机权重都会是一样，
                # 否则，因为不同的权重随机初始化会导致different random weight initializations can lead to different validation accuracy.
                random_state=1\
                )

######################
# 开始训练
scores = []
for train_indices, test_indices in cv.split(X):
    # clf.fit: Fit the model according to the given training data.
    # y[train_indices]因为y的数据结构原因得到的数据是[[1] [0] [1]...], 需要ravel变成flatten成[1 0 1...]
    classifier.fit(X[train_indices], y[train_indices].ravel())

    # 自主添加
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # clf.score: 针对这类Classifier的结果评价使用的是[分类正确的样本数量/总数量]来衡量
    # Returns the mean accuracy on the given test data and labels.
    # Each classifier provides it's own scoring function.
    # ravel() return a contiguous flattened array.
    scores.append( classifier.score(X_test, y_test.ravel()) )

min_score = min(scores)
max_score = max(scores)
avg_score = sum(scores) / len(scores)

# get_params()获得classifier的初始化参数信息
print("Classifier info:\n", classifier.get_params())
# Score代表[分类正确的样本数量/总数量]
p_score = lambda model, score: print('Performance of the %s model is %0.3f%%' % (model, score * 100))
p_score("MPLClassifier", avg_score)
# 如果只有1个hidden layer，那么加上input layer和output layer，n_layers_=3
print("Num of hidden layer:", classifier.n_layers_-2)
# Number of outputs
print("Num of outputs:", classifier.n_outputs_)
# scikit-learn源代码中是这样设置out_activation_
# 如果是Output for regression，那么out_activation_ = 'identity'，
# 如果是Output for multi class，那么out_activation_ = 'softmax'，
# 如果是Output for binary class and multi-label，那么out_activation_ = 'logistic'
# python中的multi-class指的是给某个对象分一个类别，multi-label则给某个对象贴一个或者多个标签
print("Output activation function:", classifier.out_activation_)
# coefs_是一个list，包含神经网络的各个权重，
# 其结构是[array1 for hidden layer1, array2 for hidden layer2...arrayn for output layer],
# 每个array的结构都是按照前一个layer的nodes个数为行，后一个layer的nodes个数为列
# print("Weights:\n", classifier.coefs_)

###################################
# predict using trained classifier
X_predict=X
y_predict=y.flatten()
y_output=classifier.predict(X_predict)
# print("Predicted output:\n",y_output)
# print("Desired output:\n",y_predict)
# print("Comparison:\n",np.subtract(y_predict,y_output))
# 获取某个array里面符合条件的元素个数
# len(np.where(np.subtract(y_predict,y_output)!=0)[0])
num_wrong_prediction=(np.subtract(y_predict,y_output)!=0).sum()
print("Num of incorrect predictions:", num_wrong_prediction, ", percetage:", num_wrong_prediction/len(y_predict))

###################################
# serialize our model and re-use it in production applications.
# pickle is used for serializing and de-serializing a Python object structure.
# import pickle
# data_path = os.path.join(os.getcwd(), "best-titanic-model.pkl")
# pickle.dump(classifier, open(data_path, 'wb'))
