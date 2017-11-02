import sklearn.datasets as dataset
import cdt
import pandas as pd
import numpy as np
from sklearn import metrics

cdt.SETTINGS.verbose = 0

nb_run = 20
n_features = 20
n_samples = 500
noise = 0


test = []
for j in range(n_features):
    if(j < 5):
        test.append(1)
    else:
        test.append(0)


noise_list = [0.0,0.1,0.2,0.5,1,2,5,10]


array_results = np.zeros((8, len(noise_list)))


for n in range(len(noise_list)):

    noise = noise_list[n]

    aupr_fsgnn_avg = 0
    aupr_hsic_avg = 0
    aupr_rrelief_avg = 0
    aupr_decisionTree_avg = 0
    aupr_linearSVR_avg = 0
    aupr_randomizedLasso_avg = 0
    aupr_ard_avg = 0
    aupr_rfecv_avg = 0

    for i in range(nb_run):
        
        print("num_run " + str(i))
        print("noise " + str(noise))

        X, y = dataset.make_friedman1(n_samples=n_samples, n_features=n_features, noise=noise, random_state=None)


        #X = np.random.uniform(0,1,(n_samples,n_features)) 
        #y = 10 * X[:, 0] * X[:, 1] + (20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]) * noise * np.random.randn(500) + noise * np.random.randn(500)



        df_features = pd.DataFrame(X, columns = list("X" + str(i) for i in range(n_features)))
        df_target = pd.DataFrame(y, columns = ["Y"])

        HSICLasso = cdt.independence.graph.HSICLasso()
        results = HSICLasso.predict_features(df_features, df_target, 10)

        aupr_hsic = metrics.average_precision_score(np.array(test) == 1, np.array(results))
        aupr_hsic_avg += aupr_hsic
        print("aupr_hsic " + str(aupr_hsic))


        RRelief = cdt.independence.graph.RRelief()
        results = RRelief.predict_features(df_features, df_target)

        aupr_rrelief = metrics.average_precision_score(np.array(test) == 1, np.array(results))
        print("aupr_rrelief " + str(aupr_rrelief))
        aupr_rrelief_avg += aupr_rrelief

        DecisionTree_regressor = cdt.independence.graph.DecisionTree_regressor()
        results = DecisionTree_regressor.predict_features(df_features, df_target)

        aupr_decisionTree = metrics.average_precision_score(np.array(test) == 1, np.array(results))
        print("aupr_decisionTree " + str(aupr_decisionTree))
        aupr_decisionTree_avg += aupr_decisionTree


        ARD_Regression = cdt.independence.graph.ARD_Regression()
        results = ARD_Regression.predict_features(df_features, df_target)
    
        aupr_ard = metrics.average_precision_score(np.array(test) == 1, np.array(results))
        print("aupr_ard " + str(aupr_ard))
        aupr_ard_avg += aupr_ard


        RFECV_linearSVR = cdt.independence.graph.RFECV_linearSVR()
        results = RFECV_linearSVR.predict_features(df_features, df_target)

        aupr_rfecv = metrics.average_precision_score(np.array(test) == 1, np.array(results))
        print("aupr_rfecv " + str(aupr_rfecv))
        aupr_rfecv_avg += aupr_rfecv

        LinearSVR_L2 = cdt.independence.graph.LinearSVR_L2()
        results = LinearSVR_L2.predict_features(df_features, df_target)

        aupr_linearSVR = metrics.average_precision_score(np.array(test) == 1, np.array(results))

        print("aupr_linearSVR L2 " + str(aupr_linearSVR))
        aupr_linearSVR_avg += aupr_linearSVR


        RandomizedLasso_model = cdt.independence.graph.RandomizedLasso_model()
        results = RandomizedLasso_model.predict_features(df_features, df_target)

        aupr_randomizedLasso = metrics.average_precision_score(np.array(test) == 1, np.array(results))
        print("aupr_randomizedLasso " + str(aupr_randomizedLasso))
        aupr_randomizedLasso_avg += aupr_randomizedLasso

        FSGNN = cdt.independence.graph.FSGNN()
        results = FSGNN.run_FS(pd.concat([df_features,df_target],1), 0, "Y")

        aupr_fsgnn = metrics.average_precision_score(np.array(test) == 1, np.array(results))
        print("aupr_fsgnn " + str(aupr_fsgnn))
        aupr_fsgnn_avg += aupr_fsgnn
     
        print("aupr_avg HSIC lasso : " + str(aupr_hsic_avg/(i+1)))
        print("aupr_avg RRelief : " + str(aupr_rrelief_avg/(i+1)))
        print("aupr_avg decisionTree : " + str(aupr_decisionTree_avg/(i+1)))
        print("aupr_avg ARD : " + str(aupr_ard_avg/(i+1)))
        print("aupr_avg rfecv : " + str(aupr_rfecv_avg/(i+1)))
        print("aupr_avg linearSVR : " + str(aupr_linearSVR_avg/(i+1)))
        print("aupr_avg randomizedLasso : " + str(aupr_randomizedLasso_avg/(i+1)))
        print("aupr_avg FSGNN : " + str(aupr_fsgnn_avg/(i+1)))

        array_results[0,n] = aupr_hsic_avg/(i+1)
        array_results[1,n] = aupr_rrelief_avg/(i+1)
        array_results[2,n] = aupr_decisionTree_avg/(i+1)
        array_results[3,n] = aupr_ard_avg/(i+1)
        array_results[4,n] = aupr_rfecv_avg/(i+1)
        array_results[5,n] = aupr_linearSVR_avg/(i+1)
        array_results[6,n] = aupr_randomizedLasso_avg/(i+1)
        array_results[7,n] = aupr_fsgnn_avg/(i+1)

        np.savetxt("results_feature_selection.csv", array_results)



   





















