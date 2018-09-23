import os
import pandas as pd
# from pandas import read_csv, concat

# 读取数据 Load data
data_path = os.path.join(os.getcwd(), "machine-learning\\data\\titanic-train.csv")
dataset = pd.read_csv(data_path, skipinitialspace=True)

dataset.columns.values
dataset.head(5)

###########################
# 对数据进行一些preprocessing：丢弃一些不用的字段
import pandas as pd

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
X = dataset.drop(['Survived'], axis = 1).values
y = dataset[['Survived']].values

# 对X进行标准化
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
p_score = lambda model, score: print('Performance of the %s model is %0.2f%%' % (model, score * 100))

# Classifiers
names = [
    "Log Regression", "Log Regression + Polynomial",
    "Linear SVM", "RBF SVM", "Neural Net",
]

classifiers = [
    LogisticRegression(),
    make_pipeline(PolynomialFeatures(3), LogisticRegression()),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    # a multi-layer perceptron (MLP) algorithm, alpha defines L2 penalty (regularization term) parameter.
    # para: hidden_layer_sizes=(100,) if you want only 1 hidden layer with 100 hidden units.
    MLPClassifier(alpha=1),
]

######################
# 开始训练
# iterate over classifiers
models_eval = []
trained_classifiers = []
# zip会对两个list对应位置的元素进行iterate. zip- Iterate over two lists in parallel
for name, clf in zip(names, classifiers):
    scores = []
    i_split = 0
    # k-fold交叉检验遍历。kfold每次split(5次)会返回training set indices & test set indices for that split.
    for train_indices, test_indices in cv.split(X):
        # clf.fit: Fit the model according to the given training data.
        # y[train_indices]因为y的数据结构原因得到的数据是[[1] [0] [1]...], 需要ravel变成flatten成[1 0 1...]
        clf.fit(X[train_indices], y[train_indices].ravel())

        # 自主添加
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # clf.score: Returns the mean accuracy on the given test data and labels.
        # Each classifier provides it's own scoring function.
        scores.append( clf.score(X_test, y_test.ravel()) )
    
    min_score = min(scores)
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    
    trained_classifiers.append(clf)
    models_eval.append((name, min_score, max_score, avg_score))
    
fin_models = pd.DataFrame(models_eval, columns = ['Name', 'Min Score', 'Max Score', 'Mean Score'])

fin_models.sort_values(['Mean Score']).head()

###################################
# serialize our model and re-use it in production applications.
# pickle is used for serializing and de-serializing a Python object structure.
import pickle

best_model = trained_classifiers[4]
# coefs_是一个list，包含神经网络的各个权重，
# 其结构是[array1 for hidden layer1, array2 for hidden layer2...arrayn for output layer],
# 每个array的结构都是按照前一个layer的nodes个数为行，后一个layer的nodes个数为列
print(best_model.coefs_)
# 如果只有1个hidden layer，那么加上input layer和output layer，n_layers_=3
print(best_model.n_layers_)
# Number of outputs
print(best_model.n_outputs_)
# scikit-learn源代码中是这样设置out_activation_
# 如果是Output for regression，那么out_activation_ = 'identity'，
# 如果是Output for multi class，那么out_activation_ = 'softmax'，
# 如果是Output for binary class and multi-label，那么out_activation_ = 'logistic'
# python中的multi-class指的是给某个对象分一个类别，multi-label则给某个对象贴一个或者多个标签
print(best_model.out_activation_)

data_path = os.path.join(os.getcwd(), "best-titanic-model.pkl")
pickle.dump(best_model, open(data_path, 'wb'))

