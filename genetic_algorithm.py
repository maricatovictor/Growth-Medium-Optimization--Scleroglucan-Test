import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from gplearn.genetic import SymbolicRegressor
import pickle
import scipy.optimize as sp

df = pd.read_excel('train_sheet.xlsx', usecols=[
                   0, 1, 2, 3], skiprows=[0], header=None)
d = df.values
l = pd.read_excel('train_sheet.xlsx', usecols=[4], skiprows=[0], header=None)
labels = l.values

#X = np.asmatrix([[35, 1, 1.25, 0.5], [65, 1, 1.25, 0.5], [35, 2, 1.75, 0.5]])
#y = np.asmatrix([[7.97], [15.22], [9.81]])

X = np.asmatrix(d)
y = np.asmatrix(labels)

tfeats = pd.read_excel('predict_sheet.xlsx', usecols=[
                       0, 1, 2, 3], skiprows=[0], header=None)
test_feature = np.asmatrix(tfeats)

import os
prev_gen = 'gp_model.pkl'
if os.path.isfile(prev_gen) and os.path.getsize(prev_gen) > 0:
    with open(prev_gen, 'rb') as f:
        #est_gp = SymbolicRegressor(population_size=100,
        #                  generations=750, stopping_criteria=0.01,
        #                   p_crossover=0.8, p_subtree_mutation=0.09,
        #                    p_hoist_mutation=0.01, p_point_mutation=0.05,
        #                     max_samples=1.0, verbose=1,
        #                      parsimony_coefficient=0.001, random_state=0)

        #est_gp.fit(X, y)

        #good:  est_gp = SymbolicRegressor(population_size=80,
        #                   generations=1000,
        #                  p_crossover=0.92, p_subtree_mutation=0.05,
        #                     verbose=1, max_samples=1,
        #                    parsimony_coefficient=0.0001)
        est_gp = pickle.load(f)
        pred = est_gp.predict(test_feature)
        print(pred)
else:
    with open('gp_model.pkl', 'wb') as f:
        #   'Better: est_gp = SymbolicRegressor(population_size=150, tournament_size=15,
        #                      generations=5000,
        #                     p_crossover=0.9, p_subtree_mutation=0.05, p_hoist_mutation=0.01, p_point_mutation=0.01,
        #                        verbose=1, max_samples=1,
        #                       parsimony_coefficient=0.0001)'
        est_gp = SymbolicRegressor(population_size=2000, tournament_size=85, stopping_criteria=0.01, const_range=(40,45),
            generations=400,
        p_crossover=0.925, p_subtree_mutation=0.05, p_hoist_mutation=0.00, p_point_mutation=0.00,
        verbose=1, max_samples=1,
        parsimony_coefficient=0.0001)

        est_gp.fit(X,y)
        pickle.dump(est_gp, f)
        pred = est_gp.predict(test_feature)
        print(pred)

import xlsxwriter as xlsw
dfw = pd.DataFrame(pred)
dfw = dfw.transpose()
xlsxfile = "output_ga.xlsx"
writer = pd.ExcelWriter(xlsxfile, engine='xlsxwriter')
dfw.to_excel(writer, sheet_name="output1", startrow=1, startcol=1, header=False, index=False)
writer.close()