from sklearn.base import BaseEstimator
import sklearn.metrics
import sys
import numpy
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import seaborn as sns

class SelfLearningModel(BaseEstimator):
    """
    Self Learning framework for semi-supervised learning

    This class takes a base model (any scikit learn estimator),
    trains it on the labeled examples, and then iteratively 
    labeles the unlabeled examples with the trained model and then 
    re-trains it using the confidently self-labeled instances 
    (those with above-threshold probability) until convergence.

    Parameters
    ----------
    basemodel : BaseEstimator instance
        Base model to be iteratively self trained

    max_iter : int, optional (default=200)
        Maximum number of iterations

    prob_threshold : float, optional (default=0.8)
        Probability threshold for self-labeled instances
    """
    
    def __init__(self, basemodel, max_iter = 200, prob_threshold = 0.8):
        self.model = basemodel
        self.model = svm.SVC(kernel='rbf', decision_function_shape='ovr', probability=True)
        self.max_iter = max_iter
        self.prob_threshold = prob_threshold 
        self.model2 = GaussianNB()
    def fit2(self, X, y): # -1 for unlabeled
        """Fit base model to the data in a semi-supervised fashion 
        using self training 

        All the input data is provided matrix X (labeled and unlabeled)
        and corresponding label matrix y with a dedicated marker value (-1) for
        unlabeled samples.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            A {n_samples by n_samples} size matrix will be created from this

        y : array_like, shape = [n_samples]
            n_labeled_samples (unlabeled points are marked as -1)

        Returns
        -------
        self : returns an instance of self.
        """
        unlabeledX = X[y==-1, :]
        labeledX = X[y!=-1, :]
        labeledy = y[y!=-1]
        
        self.model.fit(labeledX, labeledy)

        unlabeledy = self.predict(unlabeledX)
        unlabeledprob = self.predict_proba(unlabeledX)
        unlabeledy_old = []
        #re-train, labeling unlabeled instances with model predictions, until convergence
        i = 0
        while (len(unlabeledy_old) == 0 or numpy.any(unlabeledy!=unlabeledy_old)) and i < self.max_iter:
            unlabeledy_old = numpy.copy(unlabeledy)
            uidx = numpy.where((unlabeledprob[:, 0] > self.prob_threshold) | (unlabeledprob[:, 1] > self.prob_threshold))[0]
            
            self.model.fit(numpy.vstack((labeledX, unlabeledX[uidx, :])), numpy.hstack((labeledy, unlabeledy_old[uidx])))
            unlabeledy = self.predict(unlabeledX)
            unlabeledprob = self.predict_proba(unlabeledX)
            i += 1
        
        if not getattr(self.model, "predict_proba", None):
            # Platt scaling if the model cannot generate predictions itself
            self.plattlr = LR()
            preds = self.model.predict(labeledX)
            self.plattlr.fit( preds.reshape( -1, 1 ), labeledy )
            
        return self

    def fit(self, X, y, **fit_param):  # -1 for unlabeled
        """Fit base model to the data in a semi-supervised fashion
        using self training

        All the input data is provided matrix X (labeled and unlabeled)
        and corresponding label matrix y with a dedicated marker value (-1) for
        unlabeled samples.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            A {n_samples by n_samples} size matrix will be created from this

        y : array_like, shape = [n_samples]
            n_labeled_samples (unlabeled points are marked as -1)

        Returns
        -------
        self : returns an instance of self.
        """
        unlabeledX = X[y == -1, :]
        labeledX = X[y != -1, :]
        labeledy = y[y != -1]
        #unlabeledY = ytrue[y == -1]

        self.model.fit(labeledX, labeledy, **fit_param)
        y_self = self.model.predict(labeledX)
        y_selfprob = self.model.predict_proba(labeledX)
        #print("labeleled sample before outlier removal: ", len(labeledX))
        #print("label accuracy only labelled :", sklearn.metrics.accuracy_score(labeledy, y_self, sample_weight=None))
        # remove outlier sample from model
        #print("outlier sample prob", y_selfprob[labeledy != y_self])
        labeledX = labeledX[labeledy == y_self]
        labeledy = labeledy[labeledy == y_self]
        #print("labeleled sample after outlier removal: ", len(labeledX), len(labeledy))
        #model2 = DecisionTreeClassifier()
        #model2 = LogisticRegression()
        model2 = GaussianNB()
        #model2 = svm.SVC(kernel='rbf', decision_function_shape='ovr', probability=True)
        model2.fit(labeledX, labeledy)
        batch_perc = 10
        batchsize = len(unlabeledX) * batch_perc // 100
        iteration_perc = 50
        start = 0
        end = batchsize
        it = 0
        x1 = numpy.copy(labeledX)
        y1 = numpy.copy(labeledy)
        #y2 = numpy.copy(labeledy) # just for debugging
        #print(len(unlabeledX), batchsize)
        #print("sample print: ", unlabeledX[0:1,:], unlabeledX[0])
        while (it < 100 and end < len(unlabeledX) * iteration_perc // 100):
            #print("loop: ", it, start, end)
            #print("loop label accuracy using model2:",
            #      sklearn.metrics.accuracy_score(y2, model2.predict(x1), sample_weight=None))
            s = unlabeledX[start:end]
            #x1 = numpy.append(x1, s, axis=0)
            #y2 = numpy.append(y2, unlabeledY[start:end], axis=0)
            try1 = model2.predict(s)
            try1_prob = model2.predict_proba(s)
            #print(len(s))
            for i in range(0, len(s)):
                if (try1_prob[i,try1[i]] > .90):
                    y1 = numpy.append(y1, try1[i:i+1], axis=0)
                    x1 = numpy.append(x1, s[i:i+1,:], axis=0)
            #y1 = numpy.append(y1, model2.predict(s), axis=0)
            #print(len(y1))
            model2.fit(x1, y1)
            start += batchsize
            end += batchsize
            it += 1
        self.model.fit(x1,y1)
        #print("Model completed:")
        if not getattr(self.model, "predict_proba", None):
            # Platt scaling if the model cannot generate predictions itself
            self.plattlr = LR()
            preds = self.model.predict(labeledX)
            self.plattlr.fit(preds.reshape(-1, 1), labeledy)

        return self

    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        
        if getattr(self.model, "predict_proba", None):
            return self.model.predict_proba(X)
        else:
            preds = self.model.predict(X)
            return self.plattlr.predict_proba(preds.reshape( -1, 1 ))
        
    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Class labels for samples in X.
        """
        
        return self.model.predict(X)
    
    def score(self, X, y, sample_weight=None):
        return sklearn.metrics.accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    