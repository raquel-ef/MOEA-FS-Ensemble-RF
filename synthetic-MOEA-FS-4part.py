import numpy as np
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from time import process_time
import pickle
import random
from utils import split_features_target, lags, predictionsModels, predictionshstepsaheadEnsemble
from MultiObjectiveRFWrapper import *

import warnings
warnings.filterwarnings("ignore")

SEED_VALUE = 0
N_STEPS = 3

df = pd.read_csv('./datasets/' + sys.argv[1] + '.csv')

# Load database and split in train, validation and test
synthetic = lags(df, N_STEPS)[N_STEPS:].reset_index(drop=True)

train, test = np.split(synthetic, [int(.8*len(synthetic))])
test = test.reset_index(drop=True)

# Standarize sets
scaler = preprocessing.StandardScaler().fit(train)
train = pd.DataFrame(scaler.transform(train), columns=train.columns)
test = pd.DataFrame(scaler.transform(test), columns=test.columns)

splitTrain = np.array_split(train, 4)
train1 = splitTrain[0]
train2 = splitTrain[1].reset_index(drop=True)
train3 = splitTrain[2].reset_index(drop=True)
train4 = splitTrain[3].reset_index(drop=True)
# train5 = splitTrain[4].reset_index(drop=True)

trainXnorm, trainYnorm = split_features_target(train)
testXnorm, testYnorm = split_features_target(test)

train1Xnorm, train1Ynorm = split_features_target(train1)
train2Xnorm, train2Ynorm = split_features_target(train2)
train3Xnorm, train3Ynorm = split_features_target(train3)
train4Xnorm, train4Ynorm = split_features_target(train4)


n_inputs = len(train1Xnorm.columns)
n_obj = 4

### Evolutionary algorithm

if __name__ == "__main__":
    results = []
    n_var = n_inputs
    nfe = 100000 
    n_seeds = 10
    n_jobs = 8

    def callback_function(algorithm):
        # Increment the evaluation counter
        evaluation_counter[0] += algorithm.population_size
        
        if evaluation_counter[0] % 5000 == 0: # to have 20 points for hypervolume evolution
            # Store the current results
            intermediate_results[evaluation_counter[0]] = unique(nondominated(algorithm.result))
    
    # define the problem definition
    problem = MultiObjectiveRFWrapper(n_var, n_obj, 
                                      *[train1Xnorm], *[train1Ynorm], 
                                      *[train2Xnorm], *[train2Ynorm], 
                                      *[train3Xnorm], *[train3Ynorm], 
                                      *[train4Xnorm], *[train4Ynorm])#, *[train5Xnorm], *[train5Ynorm])

    # instantiate the optimization algorithm to run in parallel      
    for s in range(n_seeds):
        intermediate_results = {} # Global dictionary to store intermediate results
        evaluation_counter = [0]
        listAttributes = []
        listTime = []
        start_time = process_time()
        print("--- Run %s ---" % s)
        random.seed(s)

        with ProcessPoolEvaluator(n_jobs) as evaluator:
            algorithm = NSGAIII(problem, divisions_outer=5, evaluator=evaluator)
            algorithm.run(nfe, callback=callback_function)
            
        # Save
        # with open('./results_embedded/' + sys.argv[1] + '-intermediate-results-MOEA-' + str(s) + '-std.pickle', 'wb') as f:
        #     pickle.dump([intermediate_results], f)

        results.append(algorithm.result)
        finaltime = np.round(((process_time() - start_time)/60), 2)
        listTime.append(finaltime)
        print("--- %s minutes ---" % finaltime)

    print("Mean time: ", np.mean(listTime))

    # Save
    with open('./results_embedded/' + sys.argv[1] + '-results-MOEA-4part-std.pickle', 'wb') as f:
        pickle.dump([results], f)


    ### Ensemble model
    listMeanTrainRF, listMeanTestRF, listMeanAttribRF = [], [], []
    listdfResultsTrainRF, listdfResultsTestRF, listpredTrainRF, listpredTestRF = [], [], [], []
    listAttributesRF = []
    listTuplesRF = []

    print("Ensemble RF")
    for r in results:
        print("------------------- RUN -------------------")
        predTrain1 = predictionsModels(r, train1Xnorm, train1Ynorm)
        predTrain2 = predictionsModels(r, train2Xnorm, train2Ynorm)
        predTrain3 = predictionsModels(r, train3Xnorm, train3Ynorm)
        predTrain4 = predictionsModels(r, train4Xnorm, train4Ynorm)
        # predTrain5 = predictionsModels(r, train5Xnorm, train5Ynorm)
        predModel = pd.concat([predTrain1, predTrain2, predTrain3, predTrain4], ignore_index=True) #, predTrain5], ignore_index=True)

        # Meta-model 
        random.seed(SEED_VALUE)
        metaModel = RandomForestRegressor(n_estimators=100, random_state=SEED_VALUE).fit(predModel.iloc[:,:-1], predModel.iloc[:,-1])

        dfResultsTrain, predTrain, _ = predictionshstepsaheadEnsemble(trainXnorm, trainYnorm, r, metaModel, N_STEPS)
        dfResultsTest, predTest, _ = predictionshstepsaheadEnsemble(testXnorm, testYnorm, r, metaModel, N_STEPS)

        listdfResultsTrainRF.append(dfResultsTrain)
        listdfResultsTestRF.append(dfResultsTest)
        listpredTrainRF.append(predTrain)
        listpredTestRF.append(predTest)

        print("Mean Train: ", np.round(np.mean(dfResultsTrain['RMSE'][1:]), 5))
        print(dfResultsTrain)

        print("Mean Test: ", np.round(np.mean(dfResultsTest['RMSE'][1:]), 5))
        print(dfResultsTest)

        listMeanTrainRF.append(np.mean(dfResultsTrain['RMSE'][1:])) 
        listMeanTestRF.append(np.mean(dfResultsTest['RMSE'][1:]))

        results = unique(nondominated(r))

        for r in results:
            listAttributesRF.append(np.array([int(x[0]) for x in r.variables[:n_inputs]]))

        # Mean selected attributes
        listMeanAttribRF.append(np.round(np.mean(np.sum(listAttributesRF, axis=1)), 2))

        cols = []
        for i in train.columns:
            cols.append(i.replace('_', '\_'))

        # Attribute importance
        tuples = [ (x,y) for x, y in zip(cols, np.round(np.sum(listAttributesRF, axis=0)/n_inputs, 2))]
        tuples.sort(key = lambda i:i[1])
        print(*tuples,sep='\n')

        listTuplesRF.append(tuples)

    dfSummaryRuns = pd.DataFrame({'Mean Train': listMeanTrainRF, 'Mean Test': listMeanTestRF, 'Mean N attrib': listMeanAttribRF})
    print(dfSummaryRuns)

    # Save
    with open('./results_embedded/' + sys.argv[1] + '-results-list-RF-4part-std.pickle', 'wb') as f:
        pickle.dump([listdfResultsTrainRF, listdfResultsTestRF, listpredTrainRF, listpredTestRF, listTuplesRF], f)
