import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from gplearn.genetic import SymbolicRegressor
import pickle
import scipy.optimize as sp
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('train_sheet.xlsx', usecols=[
                   0, 1, 2, 3], skiprows=[0], header=None)
d = df.values
l = pd.read_excel('train_sheet.xlsx', usecols=[4], skiprows=[0], header=None)
labels = l.values

#X = np.asmatrix([[35, 1, 1.25, 0.5], [65, 1, 1.25, 0.5], [35, 2, 1.75, 0.5]])
#y = np.asmatrix([[7.97], [15.22], [9.81]])

X = np.asmatrix(d)
y = np.asmatrix(labels)

scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X)


tfeats = pd.read_excel('predict_sheet.xlsx', usecols=[
                       0, 1, 2, 3], skiprows=[0], header=None)
test_feature = np.asmatrix(tfeats)

# Now apply the transformations to the data:
X = scaler.transform(X)
test_feature = scaler.transform(test_feature)

#MLPRegressor

clf = MLPRegressor(
  hidden_layer_sizes=(6,6,6,6),
    activation='logistic',
    solver='lbfgs',
    alpha=0.00001,
    batch_size='auto',
    verbose=True,
    random_state=None)

clf.fit(X, y)

ann_predicted = clf.predict(test_feature)

print(ann_predicted)


import xlsxwriter as xlsw
dfw = pd.DataFrame(ann_predicted)
dfw = dfw.transpose()
xlsxfile = "output_ann.xlsx"
writer = pd.ExcelWriter(xlsxfile, engine='xlsxwriter')
dfw.to_excel(writer, sheet_name="output1", startrow=1,
             startcol=1, header=False, index=False)
writer.close()
