import sklearn.datasets as dataset
import cdt
import pandas as pd

X, y = dataset.make_friedman1(n_samples=500, n_features=10, noise=1.0, random_state=None)

df_features = pd.DataFrame(X)
df_target = pd.DataFrame(y)


# HSICLasso = cdt.independence.graph.HSICLasso()
# results = HSICLasso.predict_features(df_features, df_target,10)
# print(results)

# DecisionTree_regressor = cdt.independence.graph.DecisionTree_regressor()
# results = DecisionTree_regressor.predict_features(df_features, df_target)
# print(results)

# ARD_Regression = cdt.independence.graph.ARD_Regression()
# results = ARD_Regression.predict_features(df_features, df_target)
# print(results)

# RRelief = cdt.independence.graph.RRelief()
# results = RRelief.predict_features(df_features, df_target)
# print(results)

# RFECV_linearSVR = cdt.independence.graph.RFECV_linearSVR()
# results = RFECV_linearSVR.predict_features(df_features, df_target)
# print(results)

# LinearSVR_L2 = cdt.independence.graph.LinearSVR_L2()
# results = LinearSVR_L2.predict_features(df_features, df_target)
# print(results)


RandomizedLasso_model = cdt.independence.graph.RandomizedLasso_model()
results = RandomizedLasso_model.predict_features(df_features, df_target)
print(results)