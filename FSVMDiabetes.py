import numpy as np
import numpy
import pylab as pl
from sklearn.model_selection import cross_val_score
import seaborn as sns
import sklearn.svm
from sklearn import svm, cross_validation
from SelfLearning import SelfLearningModel
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from pandas import read_csv
from tkinter import *
from mice import *
from CPLELearning import CPLELearningModel
from scikitWQDA import WQDA
from scikitTSVM import SKTSVM
#import pomegranate
#from pomegranate import NaiveBayes

import warnings
warnings.filterwarnings('ignore')
label_sample_perc = 20

class DiabetesPrediction:

    def __init__(self, data = "diabetes"):
        self.data = data

    def data_processing(self, fileName='pima-indians-diabetes.csv'):
        dataset = read_csv(fileName, header=None)
        #dataset = fetch_mldata(self.data)

        # replace zero with mean value for few colunms
        dataset[[1, 2, 3, 4, 5]] = dataset[[1, 2, 3, 4, 5]].replace(0, numpy.NaN)
        values = dataset.values
        imputer = MICE(n_imputations=100, impute_type='pmm', n_nearest_columns=5, verbose= FALSE)
        transformed_values = imputer.complete(values)
        X = transformed_values[:, 0:8]
        ytrue = transformed_values[:, 8]
        # feature selection
        X = X[:, [0, 1, 2, 5, 6, 7]]
        sc_X = StandardScaler()
        X = sc_X.fit_transform(X)
        return X, ytrue, sc_X

    def unlabel_data(self, ytrue, seed = 42, label_perc = .2):
        # split label and unlabeled data
        rng = np.random.RandomState(seed)
        random_labeled_points = rng.rand(len(ytrue)) < label_perc
        ys = np.array([-1] * len(ytrue))  # -1 denotes unlabeled point
        #label_perc = label_sample_perc
        #label_len = len(ytrue) * label_perc // 100
        #for x in range(0, label_len):
        #    ys[x] = ytrue[x]
        ys[random_labeled_points] = ytrue[random_labeled_points]
        return ys

    def validation(self, y_test, y_pred_test, y_pred_prob):
        acc = sklearn.metrics.accuracy_score(y_test, y_pred_test, sample_weight=None)
        print("Accuracy:", acc)
        print("F1 SCORE: ", f1_score(y_test, y_pred_test))
        print("classification report: ")
        print(classification_report(y_test, y_pred_test))
        cm = confusion_matrix(y_test, y_pred_test)
        TP = cm[1, 1]
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        classification_error = (FP + FN) / float(TP + TN + FP + FN)
        print("classification_error: ", classification_error)
        sensitivity = TP / float(FN + TP)
        print("sensitivity: ", sensitivity) # also known as recall score, When the actual value is positive, how often is the prediction correct?
        specificity = TN / (TN + FP)
        print("specificity: ", specificity) # When the actual value is negative, how often is the prediction correct?
        precision = TP / float(TP + FP)
        print("precision: ", precision) # How "precise" is the classifier when predicting positive instances?
        roc_auc = sklearn.metrics.roc_auc_score(y_test, y_pred_prob)
        print("ROC Curve AUC Area: ", roc_auc)
        print("Confusion matrix:")
        print(cm)
        label = ["0", "1"]
        sns.heatmap(cm, annot=True, xticklabels=label, yticklabels=label)
        plt.show()
        # plot histogram of predicted probability of diabtes
        plt.rcParams['font.size'] = 12
        # 8 bins
        plt.hist(y_pred_prob, bins=8)
        # x-axis limit from 0 to 1
        plt.xlim(0, 1)
        plt.title('Histogram of predicted probabilities')
        plt.xlabel('Predicted probability of diabetes')
        plt.ylabel('Frequency')
        plt.show()
        # plot ROC curve
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_pred_prob)
        print("fpr below")
        print(fpr)
        print("tpr below")
        print(tpr)
        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve for diabetes classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.grid(True)
        plt.show()
        return acc, sensitivity, specificity, roc_auc

    def cross_valid(self, model, X, Y):
        # Constants
        num_folds = 10
        num_instances = len(X)
        seed = 42
        np.random.seed(seed)
        kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
        #kfold = cross_validation.StratifiedKFold(n_splits=num_folds, random_state=seed)
        results = cross_val_score(model, X, Y, cv=kfold)

        results *= 100.0
        info = "Model 10 fold Accuracy mean: %.2f%% (+/- %.3f%%)" % (results.mean(), results.std())
        print(info)
        #print(results)

    def cross_valid2(self, model, X, y, label_perc = .8, test_train_split = .2, show_plot=False):
        results = []
        result_mean = []
        for i in range(0,10):
            # split train, test data
            X_train, X_test, ytrue, y_test = model_selection.train_test_split(X, y, test_size=test_train_split,
                                                                        random_state=5+i)

            # split label and unlabel sample
            ys = self.unlabel_data(ytrue, 5+i, label_perc)

            model.fit(X_train, ys)
            y_pred_test = model.predict(X_test)
            y_pred_test_prob = model.predict_proba(X_test)[:, 1]
            accuracy = sklearn.metrics.accuracy_score(y_test, y_pred_test, sample_weight=None)
            results.append(accuracy *  100.0 )
        print(results)
        print("Model 10 fold Accuracy mean: %.2f%% (+/- %.3f%%)" %(np.mean(results), np.std(results)), "label %", label_perc)
        result_mean.append(np.mean(results))
        if show_plot:
            fig, ax = plt.subplots()
            plt.axis([1, 10, 0, 100])
            plt.title("10 fold CV Accuracy variance")
            sns.pointplot(x=[1,2,3,4,5,6,7,8,9,10], y=results, ax=ax, x_min = 0, x_max = 10, y_min = 0, y_max = 100)
            ax.set_xlabel('Index Number for trial')
            ax.set_ylabel('Accuracy')
            plt.show()
        return result_mean

    def validate_algo(self, X, ytrue, model):
        self.cross_valid2(model, X, ytrue, show_plot=TRUE)
        label_percs = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
        result = []
        for i in label_percs:
            result = numpy.append(result, self.cross_valid2(model, X, ytrue, i), axis=0)
        print(result)
        print("Model 10 fold Accuracy with varrying label mean: %.2f%% (+/- %.3f%%)" % (np.mean(result), np.std(result)))

        fig, ax = plt.subplots()
        plt.axis([0, 1, 0, 100])
        plt.title("10 fold CV Accuracy with label sample %")
        sns.pointplot(x=label_percs, y=result, ax=ax, x_min = 0, x_max = 1, y_min = 0, y_max = 100)
        ax.set_xlabel('Labeled Sample Percentage')
        ax.set_ylabel('Accuracy')
        plt.show()

        test_train_splits = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
        result = []
        for i in test_train_splits:
            result = numpy.append(result, self.cross_valid2(model, X, ytrue, .5, i), axis=0)
        print(result)
        print("Model 10 fold Accuracy with varrying test data mean: %.2f%% (+/- %.3f%%)" % (np.mean(result), np.std(result)))
        fig, ax = plt.subplots()
        plt.axis([0, 1, 0, 100])
        plt.title("10 fold CV Accuracy with test sample %")
        sns.pointplot(x=test_train_splits, y=result, ax=ax, x_min = 0, x_max = 1, y_min = 0, y_max = 100)
        ax.set_xlabel('Test Sample Percentage')
        ax.set_ylabel('Accuracy')
        plt.show()

    def process(self):
        X, ytrue, sc_X = self.data_processing()
        self.basemodel = svm.SVC(kernel='rbf', decision_function_shape='ovr', probability=True)


        print("SVM model cross Validation")
        # create SVM model
        self.model2 = svm.SVC(kernel='sigmoid', decision_function_shape='ovr', probability=True, gamma=.1,
                              coef0=.5)
        self.cross_valid(self.model2, X, ytrue)

        #TSVM
        print("T SVM Semi Supervised Classifier cross Validation")
        self.TSVMmodel = SKTSVM(kernel='rbf')
        #self.validate_algo(X, ytrue, self.TSVMmodel)

        #S3VMmodel
        print("CPLE SVM Semi Supervised Classifier cross Validation")
        self.S3VMmodel = CPLELearningModel(self.basemodel, predict_from_probabilities=True)  # RBF SVM
        #self.validate_algo(X, ytrue, self.S3VMmodel)
        #self.cross_valid2(self.S3VMmodel, X, ytrue, show_plot=TRUE, label_perc = .5)

        # create semi supervised model with svm as base model
        self.ssmodel = SelfLearningModel(self.basemodel)
        print("Fast Semi Supervised Classifier cross Validation")
        #self.validate_algo(X, ytrue, self.ssmodel)

        # split train, test data
        X, X_test, ytrue, y_test = model_selection.train_test_split(X, ytrue, test_size=.2, random_state=7)

        #split label and unlabel sample
        ys = self.unlabel_data(ytrue, 42, .8)

        # model with simple SVM
        self.model2.fit(X, ytrue)
        print("Simple SVM Model")
        y_pred_train_svm = self.model2.predict(X)
        y_pred_train_prob_svm = self.model2.predict_proba(X)[:, 1]
        print("SVM Algo Train Data Validation")
        self.validation(ytrue, y_pred_train_svm, y_pred_train_prob_svm)
        # test data with svm
        y_pred_test_svm = self.model2.predict(X_test)
        y2_pred_prob_svm = self.model2.predict_proba(X_test)[:, 1]
        print("SVM Algo Test Data Validation")
        self.validation(y_test, y_pred_test_svm, y_pred_prob_svm)

        # fit TSVM semi supervised model
        self.TSVMmodel.fit(X, ys)
        print("TSVM Semi Supervised Fast Algo ready")
        y_pred_train = self.TSVMmodel.predict(X)
        y_pred_train_prob = self.TSVMmodel.predict_proba(X)[:, 1]
        print("TSVM Semi Supervised Fast Algo Train Data Validation")
        self.validation(ytrue, y_pred_train, y_pred_train_prob)

        y_pred_test = self.TSVMmodel.predict(X_test)
        y_pred_prob = self.TSVMmodel.predict_proba(X_test)[:, 1]
        print("TSVMmodel Semi Supervised Fast Algo Test Data Validation")
        self.validation(y_test, y_pred_test, y_pred_prob)

        # fit CPLE semi supervised model
        self.S3VMmodel.fit(X, ys)
        print("CPLE Semi Supervised Fast Algo ready")
        y_pred_train = self.S3VMmodel.predict(X)
        y_pred_train_prob = self.S3VMmodel.predict_proba(X)[:, 1]
        print("CPLE Semi Supervised Fast Algo Train Data Validation")
        self.validation(ytrue, y_pred_train, y_pred_train_prob)

        y_pred_test = self.S3VMmodel.predict(X_test)
        y_pred_prob = self.S3VMmodel.predict_proba(X_test)[:, 1]
        print("CPLE Semi Supervised Fast Algo Test Data Validation")
        self.validation(y_test, y_pred_test, y_pred_prob)

        # fit Fast semi supervised model
        self.ssmodel.fit(X, ys)
        print("Semi Supervised Fast Algo ready")
        y_pred_train = self.ssmodel.predict(X)
        y_pred_train_prob = self.ssmodel.predict_proba(X)[:, 1]
        print("Semi Supervised Fast Algo Train Data Validation")
        self.validation(ytrue, y_pred_train, y_pred_train_prob)

        y_pred_test = self.ssmodel.predict(X_test)
        y_pred_prob = self.ssmodel.predict_proba(X_test)[:, 1]
        print("Semi Supervised Fast Algo Test Data Validation")
        return self.validation(y_test, y_pred_test, y_pred_prob)

    def predict(self, x):
        return self.ssmodel.predict(x)

    def plot_boundary(self, pl, model, title):
        X1, ytrue, sc_X = self.data_processing()
        # create PCA transform
        pca = PCA(n_components=2).fit(X1)
        pca_2d = pca.transform(X1)
        for i in range(0, pca_2d.shape[0]):
            if ytrue[i] == 0:
                c1 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
            else:
                c2 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
        pl.legend([c1, c2], ['Diabetes', 'No Diabetes'])
        x_min, x_max = pca_2d[:, 0].min() - 1, pca_2d[:, 0].max() + 1
        y_min, y_max = pca_2d[:, 1].min() - 1, pca_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, .01), np.arange(y_min, y_max, .01))

        # split label and unlabeled data for PCA self learning model
        ys = self.unlabel_data(ytrue, 42, .8)

        # create self learning model for PCA
        #basemodel = svm.SVC(kernel='rbf', decision_function_shape='ovr', probability=True)
        #ssmodel = SelfLearningModel(basemodel)
        model.fit(pca_2d, ys)
        print("PCA model built")
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        SMALL_SIZE = 14
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 16
        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        pl.contour(xx, yy, Z)
        pl.axis('off')
        pl.title(title)
        pl.show()
        return pl

    def Run_Algo(self):
        # main code
        D = DiabetesPrediction()
        D.process()

        # testing
        X1, ytrue, sc_X = D.data_processing()
        ##sample = [[6, 148, 72, 33.5, 0.627, 50]]
        ##sample = sc_X.transform(sample)
        print("testing first 10 samples:")
        print("Actual Y values:", ytrue[:10])
        print("Semi Supervised predicted Y values", D.predict(X1[:10, :]))
        print("Semi supervised predicted Y prob")
        print(D.ssmodel.predict_proba(X1[:10, :]))

        # plot model decision boundary
        D.plot_boundary(plt, self.ssmodel)
        D.plot_boundary(plt, self.TSVMmodel)

# main code
D = DiabetesPrediction()
D.process()

# testing
X1, ytrue, sc_X = D.data_processing()
##sample = [[6, 148, 72, 33.5, 0.627, 50]]
##sample = sc_X.transform(sample)
print("testing first 10 samples:")
print("Actual Y values:", ytrue[:10])
print("Semi Supervised predicted Y values", D.predict(X1[:10, :]))
print("Semi supervised predicted Y prob")
print(D.ssmodel.predict_proba(X1[:10, :]))

# plot model decision boundary
#fig, ax = plt.subplots()
D.plot_boundary(plt, D.ssmodel, 'Fast SVM pima india decision boundary')
D.plot_boundary(plt, D.TSVMmodel, 'S3VM pima india decision boundary')
D.plot_boundary(plt, D.S3VMmodel, 'CPLE pima india decision boundary')
