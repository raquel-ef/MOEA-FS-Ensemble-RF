from platypus import *
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate 


class MultiObjectiveRFWrapper(Problem):

    def __init__(self, nvars, nobjs, *args):
        super(MultiObjectiveRFWrapper, self).__init__(nvars, nobjs)
        self.types[:] = Binary(1)
        self.args = args
    
    def evaluate(self, solution, *args):
        mask = []

        # Create an array 'mask' from individual values
        for sol in solution.variables:
            mask.append(sol[0])
        
        rmseList = []
        N = np.sum(mask)

        rkf = KFold(n_splits=5) 
        
        scores = {}
        for e in range(0, len(self.args), 2): # number of partitions
            if N == 0:
                predTest = [np.mean(np.asanyarray(self.args[e]))] * len(self.args[e+1])
                rmseTest = -root_mean_squared_error(self.args[e+1], predTest) 
                scores['test_neg_root_mean_squared_error'] = rmseTest     

            else:
                eTrain = self.args[e].loc[:, mask] # dataset with selected atributes

                modelRF = RandomForestRegressor(n_estimators=10, random_state=0)

                random.seed(0)
                scores = cross_validate(modelRF,
                                        eTrain, self.args[e+1], 
                                        cv=rkf, 
                                        scoring=['neg_root_mean_squared_error'],
                                        n_jobs=None,
                                        verbose=0)
            
            rmseList.append(-np.mean(scores['test_neg_root_mean_squared_error']))
          
        
        solution.objectives[:] = rmseList
        
        