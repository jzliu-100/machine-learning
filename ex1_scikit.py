import matplotlib.pyplot as plt
from pandas import read_csv
import os
from sklearn.linear_model import LinearRegression
# LinearRegression uses the gradient descent method
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

# Load data
data_path = os.path.join(os.getcwd(), "data/blood-pressure.txt")
dataset = read_csv(data_path, delim_whitespace=True)

# Our data
X = dataset[['Age']]
y = dataset[['Pressure']]

regr = LinearRegression()
regr.fit(X, y)

# Plot outputs
plt.figure(dpi=200)
plt.xlabel('Age')
plt.ylabel('Blood pressure')

plt.scatter(X, y,  color='black')
plt.plot(X, regr.predict(X), color='blue')

plt.show()
plt.gcf().clear()