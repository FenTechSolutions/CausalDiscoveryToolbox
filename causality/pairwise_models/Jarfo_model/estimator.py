"""
Cause-effect models.

"""

# Author: Jose A. R. Fonollosa <jarfo@yahoo.com>
#
# License: Apache, Version 2.0

import features as f
import numpy as np
from sklearn import pipeline
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from multiprocessing import Pool

gbc_params = {
    'loss':'deviance',
    'learning_rate': 0.1,
    'n_estimators': 500,
    'subsample': 1.0,
    'min_samples_split': 8,
    'min_samples_leaf': 1,
    'max_depth': 9,
    'init': None,
    'random_state': 1,
    'max_features': None,
    'verbose': 0
    }

selected_features = [
    'Adjusted Mutual Information[A,A type,B,B type]',
    'Conditional Distribution Entropy Variance[A,A type,B,B type]',
    'Conditional Distribution Entropy Variance[B,B type,A,A type]',
    'Conditional Distribution Kurtosis Variance[A,A type,B,B type]',
    'Conditional Distribution Kurtosis Variance[B,B type,A,A type]',
    'Conditional Distribution Similarity[A,A type,B,B type]',
    'Conditional Distribution Similarity[B,B type,A,A type]',
    'Conditional Distribution Skewness Variance[A,A type,B,B type]',
    'Conditional Distribution Skewness Variance[B,B type,A,A type]',
    'Discrete Conditional Entropy[A,A type,B,B type]',
    'Discrete Conditional Entropy[B,B type,A,A type]',
    'Discrete Entropy[A,A type]',
    'Discrete Entropy[B,B type]',
    'Discrete Mutual Information[A,A type,B,B type]',
    'HSIC[A,A type,B,B type]',
    'IGCI[A,A type,B,B type]',
    'IGCI[B,B type,A,A type]',
    'Kurtosis[A,A type]',
    'Kurtosis[B,B type]',
    'Log[Number of Samples[A]]',
    'Log[Number of Unique Samples[A]]',
    'Log[Number of Unique Samples[B]]',
    'Moment21[A,A type,B,B type]',
    'Moment21[B,B type,A,A type]',
    'Moment31[A,A type,B,B type]',
    'Moment31[B,B type,A,A type]',
    'Normalized Discrete Entropy[A,A type]',
    'Normalized Discrete Entropy[B,B type]',
    'Normalized Discrete Mutual Information[Discrete Mutual Information[A,A type,B,B type],Discrete Joint Entropy[A,A type,B,B type]]',
    'Normalized Discrete Mutual Information[Discrete Mutual Information[A,A type,B,B type],Min[Discrete Entropy[A,A type],Discrete Entropy[B,B type]]]',
    'Normalized Entropy[A,A type]',
    'Normalized Entropy[B,B type]',
    'Normalized Error Probability[A,A type,B,B type]',
    'Normalized Error Probability[B,B type,A,A type]',
#    'Number of Unique Samples[A]',
#    'Number of Unique Samples[B]',
    'Pearson R[A,A type,B,B type]',
    'Polyfit Error[A,A type,B,B type]',
    'Polyfit Error[B,B type,A,A type]',
    'Polyfit[A,A type,B,B type]',
    'Polyfit[B,B type,A,A type]',
    'Skewness[A,A type]',
    'Skewness[B,B type]',
    'Uniform Divergence[A,A type]',
    'Uniform Divergence[B,B type]'
    ]

class Pipeline(pipeline.Pipeline):
    def predict(self, X):
        try:
            p = super(Pipeline, self).predict_proba(X)
            if p.shape[1] == 2:
                p = p[:,1]
            elif p.shape[1] == 3:
                p = p[:,2] - p[:,0]
        except AttributeError:
            p = super(Pipeline, self).predict(X)
        return p

def get_pipeline(features, regressor=None, params=None):
    steps = [
        ("extract_features", f.FeatureMapper(features)),
        ("regressor", regressor(**params)),
        ]
    return Pipeline(steps)

class CauseEffectEstimatorOneStep(BaseEstimator):
    def __init__(self, features=None, regressor=None, params=None, symmetrize=True):
        self.extractor = f.extract_features
        self.classifier = get_pipeline(features, regressor, params)
        self.symmetrize = symmetrize
    
    def extract(self, features):
        return self.extractor(features)

    def fit(self, X, y=None):
        self.classifier.fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        return self.classifier.fit_transform(X, y)

    def transform(self, X):
        return self.classifier.transform(X)

    def predict(self, X):
        predictions = self.classifier.predict(X)
        if self.symmetrize:
            predictions[0::2] = (predictions[0::2] - predictions[1::2])/2
            predictions[1::2] = -predictions[0::2]
        return predictions

class CauseEffectEstimatorSymmetric(BaseEstimator):
    def __init__(self, features=None, regressor=None, params=None, symmetrize=True):
        self.extractor = f.extract_features
        self.classifier_left = get_pipeline(features, regressor, params)
        self.classifier_right = get_pipeline(features, regressor, params)
        self.symmetrize = symmetrize
    
    def extract(self, features):
        return self.extractor(features)

    def fit(self, X, y=None):
        target_left = np.array(y)
        target_left[target_left != 1] = 0
        weight_left = np.ones(len(target_left))
        weight_left[target_left==0] = sum(target_left==1)/float(sum(target_left==0))    
        try:
            self.classifier_left.fit(X, target_left, regressor__sample_weight=weight_left)
        except TypeError:
            self.classifier_left.fit(X, target_left)
        target_right = np.array(y)
        target_right[target_right != -1] = 0
        target_right[target_right == -1] = 1
        weight_right = np.ones(len(target_right))
        weight_right[target_right==0] = sum(target_right==1)/float(sum(target_right==0))        
        try:
            self.classifier_right.fit(X, target_right, regressor__sample_weight=weight_right)
        except TypeError:
            self.classifier_right.fit(X, target_right)
       
        return self

    def fit_transform(self, X, y=None):
        target_left = np.array(y)
        target_left[target_left != 1] = 0
        X_left = self.classifier_left.fit_transform(X, target_left)
        target_right = np.array(y)
        target_right[target_right != -1] = 0
        target_right[target_right == -1] = 1
        X_right = self.classifier_right.fit_transform(X, target_right)
        return X_left, X_right

    def transform(self, X):
        return self.classifier_left.transform(X), self.classifier_right.transform(X)

    def predict(self, X):
        predictions_left = self.classifier_left.predict(X)
        predictions_right = self.classifier_right.predict(X)
        predictions = predictions_left - predictions_right
        if self.symmetrize:
            predictions[0::2] = (predictions[0::2] - predictions[1::2])/2
            predictions[1::2] = -predictions[0::2]
        return predictions

class CauseEffectEstimatorID(BaseEstimator):
    def __init__(self, features_independence=None, features_direction=None, regressor=None, params=None, symmetrize=True):
        self.extractor = f.extract_features
        self.classifier_independence = get_pipeline(features_independence, regressor, params)
        self.classifier_direction = get_pipeline(features_direction, regressor, params)
        self.symmetrize = symmetrize
    
    def extract(self, features):
        return self.extractor(features)

    def fit(self, X, y=None):
        #independence training pairs
        train_independence = X
        target_independence = np.array(y)
        target_independence[target_independence != 0] = 1
        weight_independence = np.ones(len(target_independence))
        weight_independence[target_independence==0] = sum(target_independence==1)/float(sum(target_independence==0))        
        try:
            self.classifier_independence.fit(train_independence, target_independence, regressor__sample_weight=weight_independence)
        except TypeError:
            self.classifier_independence.fit(train_independence, target_independence)
        #direction training pairs
        direction_filter = y != 0
        train_direction = X[direction_filter]
        target_direction = y[direction_filter]
        weight_direction = np.ones(len(target_direction))
        weight_direction[target_direction==0] = sum(target_direction==1)/float(sum(target_direction==0))        
        try:
            self.classifier_direction.fit(train_direction, target_direction, regressor__sample_weight=weight_direction)
        except TypeError:
            self.classifier_direction.fit(train_direction, target_direction)
        return self

    def fit_transform(self, X, y=None):
        #independence training pairs
        train_independence = X
        target_independence = np.array(y)
        target_independence[target_independence != 0] = 1
        X_ind = self.classifier_independence.fit_transform(train_independence, target_independence)
        #direction training pairs
        direction_filter = y != 0
        train_direction = X[direction_filter]
        target_direction = y[direction_filter]
        self.classifier_direction.fit(train_direction, target_direction)
        X_dir = self.classifier_direction.transform(X)
        return X_ind, X_dir

    def transform(self, X):
        X_ind = self.classifier_independence.transform(X)
        X_dir = self.classifier_direction.transform(X)
        return X_ind, X_dir

    def predict(self, X):
        predictions_independence = self.classifier_independence.predict(X)
        if self.symmetrize:
            predictions_independence[0::2] = (predictions_independence[0::2] + predictions_independence[1::2])/2
            predictions_independence[1::2] = predictions_independence[0::2]
        assert predictions_independence.min() >= 0
        predictions_direction = self.classifier_direction.predict(X)
        if self.symmetrize:
            predictions_direction[0::2] = (predictions_direction[0::2] - predictions_direction[1::2])/2
            predictions_direction[1::2] = -predictions_direction[0::2]
        return predictions_independence * predictions_direction

def calculate_method(args):
    obj = args[0]
    name = args[1]
    margs = args[2]
    method = getattr(obj, name)
    return method(*margs)

def pmap(func, mlist, n_jobs):
    if n_jobs != 1:
        pool = Pool(n_jobs if n_jobs != -1 else None)
        mmap = pool.map
    else:
        mmap = map
    return mmap(func, mlist)

class CauseEffectSystemCombination(BaseEstimator):  
    def __init__(self, extractor=f.extract_features, weights=None, symmetrize=True, n_jobs=-1):
        self.extractor = extractor
        self.features = selected_features
        self.systems = [
            CauseEffectEstimatorID(
                features_direction=self.features, 
                features_independence=self.features,
                regressor=GradientBoostingClassifier,
                params=gbc_params,
                symmetrize=symmetrize), 
            CauseEffectEstimatorSymmetric(
                features=self.features,
                regressor=GradientBoostingClassifier,
                params=gbc_params,
                symmetrize=symmetrize),
            CauseEffectEstimatorOneStep(
                features=self.features,
                regressor=GradientBoostingClassifier,
                params=gbc_params,
                symmetrize=symmetrize),
        ]
        self.weights = weights
        self.n_jobs = n_jobs

    def extract(self, features):
        return self.extractor(features, n_jobs=self.n_jobs)

    def fit(self, X, y=None):
        task = [(m, 'fit', (X, y)) for m in self.systems]
        self.systems = pmap(calculate_method, task, self.n_jobs)
        return self

    def fit_transform(self, X, y=None):
        task = [(m, 'fit_transform', (X, y)) for m in self.systems]
        return pmap(calculate_method, task, self.n_jobs)

    def transform(self, X):
        task = [(m, 'transform', (X,)) for m in self.systems]
        return pmap(calculate_method, task, self.n_jobs)

    def predict(self, X):
        task = [(m, 'predict', (X,)) for m in self.systems]
        a = np.array(pmap(calculate_method, task, self.n_jobs))
        if self.weights is not None:
            return np.dot(self.weights, a)
        else:
            return a
