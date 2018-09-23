from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from pandas import read_csv
import os

# Load data
data_path = os.path.join(os.getcwd(), "data/shuttle-landing-control.data")
dataset = read_csv(data_path, header=None, 
                    names=['Auto', 'Stability', 'Error', 'Sign', 'Wind', 'Magnitude', 'Visibility'],
                    na_values='*').fillna(0)

# Prepare features
X = dataset[['Stability', 'Error', 'Sign', 'Wind', 'Magnitude', 'Visibility']]
y = dataset[['Auto']].values.reshape(1, -1)[0]

model = LogisticRegression()
model.fit(X, y)

# For now, we're missing one important concept. We don't know how well our model 
# works, and because of that, we cannot really improve the performance of our hypothesis. 
# There are a lot of useful metrics, but for now, we will validate how well 
# our algorithm performs on the dataset it learned from.
print("Score of our model is %2.2f%%" % (model.score(X, y) * 100))