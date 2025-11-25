from logging import warning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

import warnings
warnings.filterwarnings('ignore')

import pickle


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


class Classification:
    def __init__(self):
        try:
            self.df = pd.read_csv('Iris.csv')
        except Exception as e:
            e_t, e_m, e_lin = sys.exc_info()
            print(f'error from line number : {e_lin.tb_lineno} : because {e_t}:')

    def KNN_model(self):
        try:
            self.df['Species'] = self.df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
            self.X = self.df.iloc[:, :-1]
            self.y = self.df.iloc[:, -1]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                    random_state=42)
            # training data
            self.reg = KNeighborsClassifier(n_neighbors=3)
            self.reg.fit(self.X_train, self.y_train)
            self.y_train_predictions = self.reg.predict(self.X_train)
            self.training_data = pd.DataFrame()
            self.training_data = self.X_train.copy()
            self.training_data['y_train'] = self.y_train
            self.training_data['Train_predictions'] = self.y_train_predictions
            # classification report of train data
            print("Knn Algorithm Classification report:")
            print(f'train data confusion matrix:\n {confusion_matrix(self.y_train, self.y_train_predictions)}')
            print(f'train data accuracy : {accuracy_score(self.y_train, self.y_train_predictions)}')
            print(f'train  classification report : {classification_report(self.y_train, self.y_train_predictions)}')
            # testing data
            self.y_test_predictions = self.reg.predict(self.X_test)
            self.testing_data = pd.DataFrame()
            self.testing_data = self.X_test.copy()
            self.testing_data['y_train'] = self.y_test
            self.testing_data['Train_predictions'] = self.y_test_predictions
            print(f'test data confusion matrix:\n {confusion_matrix(self.y_test, self.y_test_predictions)}')
            print(f'test data accuracy : {accuracy_score(self.y_test, self.y_test_predictions)}')
            print(f'test  classification report : {classification_report(self.y_test, self.y_test_predictions)}')
            # EDA plt.figure(figsize=(5,3))
            plt.subplot(1, 2, 1)
            plt.title('k_value_train_performance')
            plt.xlabel('k_values')
            plt.ylabel('train_acc')
            plt.plot(self.y_train, self.y_train_predictions, color='r', marker='*')
            # ______________
            plt.subplot(1, 2, 2)
            plt.title('k_value_test_performance')
            plt.xlabel('k_values')
            plt.ylabel('test_acc')
            plt.plot(self.y_test, self.y_test_predictions, color='g', marker='*')
            plt.show()
            # logistic regression
            self.knn_reg = KNeighborsClassifier(n_jobs=5)
            self.test_acc_val = cross_val_score(self.knn_reg, self.X, self.y, cv=5)
            print(f'cross validation test accuracy : {np.mean(self.test_acc_val)}\n')

            # Saving PROJECT
            with open('KNN_model.pkl', 'wb') as f:
                pickle.dump(self.reg, f)

        except Exception as e:
            e_t, e_m, e_lin = sys.exc_info()
            print(f'error from line number : {e_lin.tb_lineno} : because {e_t}:')

    def Naive_bayes(self):
        try:
            self.nb_reg = GaussianNB()
            self.nb_reg.fit(self.X_train, self.y_train)
            self.N_y_train_predictions = self.nb_reg.predict(self.X_train)
            self.training_data['Train_predictions'] = self.N_y_train_predictions
            # classification report of train data
            print("Naive Bayes Algorithm Classification report:")
            print(f'train data confusion matrix:\n {confusion_matrix(self.y_train, self.N_y_train_predictions)}')
            print(f'train data accuracy : {accuracy_score(self.y_train, self.N_y_train_predictions)}')
            print(f'train  classification report : {classification_report(self.y_train, self.N_y_train_predictions)}')
            # testing data
            self.N_y_test_predictions = self.nb_reg.predict(self.X_test)
            self.testing_data['Train_predictions'] = self.N_y_test_predictions
            print(f'test data confusion matrix:\n {confusion_matrix(self.y_test, self.N_y_test_predictions)}')
            print(f'test data accuracy : {accuracy_score(self.y_test, self.N_y_test_predictions)}')
            print(f'test  classification report : {classification_report(self.y_test, self.N_y_test_predictions)}')
            # logistic regression
            self.N_reg = LogisticRegression()
            self.N_test_acc_val = cross_val_score(self.N_reg, self.X, self.y, cv=5)
            print(f'cross validation test accuracy : {np.mean(self.N_test_acc_val)}\n')

            # Saving PROJECT
            with open('Naive_bayes_model.pkl', 'wb') as f:
                pickle.dump(self.nb_reg, f)

        except Exception as e:
            e_t, e_m, e_lin = sys.exc_info()
            print(f'error from line number : {e_lin.tb_lineno} : because {e_t}:')

    def DecisionTree(self):
        try:
            self.dt_reg = DecisionTreeClassifier(criterion='entropy')
            self.dt_reg.fit(self.X_train, self.y_train)
            self.D_y_train_predictions = self.dt_reg.predict(self.X_train)
            self.training_data['Train_predictions'] = self.D_y_train_predictions
            # classification report of train data
            print("Decision Tree Algorithm Classification report:")
            print(f'train data confusion matrix:\n {confusion_matrix(self.y_train, self.D_y_train_predictions)}')
            print(f'train data accuracy : {accuracy_score(self.y_train, self.D_y_train_predictions)}')
            print(f'train  classification report : {classification_report(self.y_train, self.D_y_train_predictions)}')
            # testing data
            self.D_y_test_predictions = self.dt_reg.predict(self.X_test)
            self.testing_data['Train_predictions'] = self.D_y_test_predictions
            print(f'test data confusion matrix:\n {confusion_matrix(self.y_test, self.D_y_test_predictions)}')
            print(f'test data accuracy : {accuracy_score(self.y_test, self.D_y_test_predictions)}')
            print(f'test  classification report : {classification_report(self.y_test, self.D_y_test_predictions)}')
            # logistic regression
            self.D_reg = LogisticRegression()
            self.D_test_acc_val = cross_val_score(self.D_reg, self.X, self.y, cv=5)
            print(f'cross validation test accuracy : {np.mean(self.D_test_acc_val)}')

            # Saving PROJECT
            with open('DT_model.pkl', 'wb') as f:
                pickle.dump(self.dt_reg, f)
        except Exception as e:
            e_t, e_m, e_lin = sys.exc_info()
            print(f'error from line number : {e_lin.tb_lineno} : because {e_t}:')

    def Random_forest(self):
        try:
            self.rf_reg = RandomForestClassifier(n_estimators=11, criterion='entropy')
            self.rf_reg.fit(self.X_train, self.y_train)
            self.R_y_train_predictions = self.rf_reg.predict(self.X_train)
            self.training_data['Train_predictions'] = self.R_y_train_predictions
            # classification report of train data
            print("Random Forest Algorithm Classification report:")
            print(f'train data confusion matrix:\n {confusion_matrix(self.y_train, self.R_y_train_predictions)}')
            print(f'train data accuracy : {accuracy_score(self.y_train, self.R_y_train_predictions)}')
            print(f'train  classification report : {classification_report(self.y_train, self.R_y_train_predictions)}')
            # testing data
            self.R_y_test_predictions = self.rf_reg.predict(self.X_test)
            self.testing_data['Train_predictions'] = self.R_y_test_predictions
            print(f'test data confusion matrix:\n {confusion_matrix(self.y_test, self.R_y_test_predictions)}')
            print(f'test data accuracy : {accuracy_score(self.y_test, self.R_y_test_predictions)}')
            print(f'test  classification report : {classification_report(self.y_test, self.R_y_test_predictions)}')
            # logistic regression
            self.R_reg = LogisticRegression()
            self.R_test_acc_val = cross_val_score(self.R_reg, self.X, self.y, cv=5)
            print(f'cross validation test accuracy : {np.mean(self.R_test_acc_val)}\n')

            # Saving PROJECT
            with open('RF_model.pkl', 'wb') as f:
                pickle.dump(self.rf_reg, f)

        except Exception as e:
            e_t, e_m, e_lin = sys.exc_info()
            print(f'error from line number : {e_lin.tb_lineno} : because {e_t}:')

    def Adaboost(self):
        try:
            self.A_reg_log = LogisticRegression()
            self.ad_reg = AdaBoostClassifier(estimator=self.A_reg_log, n_estimators=11, learning_rate=0.1)
            self.ad_reg.fit(self.X_train, self.y_train)
            self.A_y_train_predictions = self.ad_reg.predict(self.X_train)
            self.training_data['Train_predictions'] = self.A_y_train_predictions
            # classification report of train data
            print("AdaBoosting Algorithm Classification report:")
            print(f'train data confusion matrix:\n {confusion_matrix(self.y_train, self.A_y_train_predictions)}')
            print(f'train data accuracy : {accuracy_score(self.y_train, self.A_y_train_predictions)}')
            print(f'train  classification report : {classification_report(self.y_train, self.A_y_train_predictions)}')
            # testing data
            self.A_y_test_predictions = self.ad_reg.predict(self.X_test)
            self.testing_data['Train_predictions'] = self.A_y_test_predictions
            print(f'test data confusion matrix:\n {confusion_matrix(self.y_test, self.A_y_test_predictions)}')
            print(f'test data accuracy : {accuracy_score(self.y_test, self.A_y_test_predictions)}')
            print(f'test  classification report : {classification_report(self.y_test, self.A_y_test_predictions)}')
            # logistic regression
            self.A_reg = LogisticRegression()
            self.A_test_acc_val = cross_val_score(self.A_reg, self.X, self.y, cv=5)
            print(f'cross validation test accuracy : {np.mean(self.A_test_acc_val)}\n')

            # Saving PROJECT
            with open('adaboost_model.pkl', 'wb') as f:
                pickle.dump(self.ad_reg, f)

        except Exception as e:
            e_t, e_m, e_lin = sys.exc_info()
            print(f'error from line number : {e_lin.tb_lineno} : because {e_t}:')

    def Gradientboosting(self):
        try:
            self.gra_reg = GradientBoostingClassifier(n_estimators=11, learning_rate=0.1)
            self.gra_reg.fit(self.X_train, self.y_train)
            self.G_y_train_predictions = self.gra_reg.predict(self.X_train)
            self.training_data['Train_predictions'] = self.G_y_train_predictions
            # classification report of train data
            print("Gradient Boosting Algorithm Classification report:")
            print(f'train data confusion matrix:\n {confusion_matrix(self.y_train, self.G_y_train_predictions)}')
            print(f'train data accuracy : {accuracy_score(self.y_train, self.G_y_train_predictions)}')
            print(f'train  classification report : {classification_report(self.y_train, self.G_y_train_predictions)}')
            # testing data
            self.G_y_test_predictions = self.gra_reg.predict(self.X_test)
            self.testing_data['Train_predictions'] = self.G_y_test_predictions
            print(f'test data confusion matrix:\n {confusion_matrix(self.y_test, self.G_y_test_predictions)}')
            print(f'test data accuracy : {accuracy_score(self.y_test, self.G_y_test_predictions)}')
            print(f'test  classification report : {classification_report(self.y_test, self.G_y_test_predictions)}')
            # logistic regression
            self.G_reg = LogisticRegression()
            self.G_test_acc_val = cross_val_score(self.G_reg, self.X, self.y, cv=5)
            print(f'cross validation test accuracy : {np.mean(self.G_test_acc_val)}\n')

            # Saving PROJECT
            with open('GB_model.pkl', 'wb') as f:
                pickle.dump(self.gra_reg, f)

        except Exception as e:
            e_t, e_m, e_lin = sys.exc_info()
            print(f'error from line number : {e_lin.tb_lineno} : because {e_t}:')

    def Xstream_boosting(self):
        try:
            self.Xgboost_reg = XGBClassifier()
            self.Xgboost_reg.fit(self.X_train, self.y_train)
            self.X_y_train_predictions = self.Xgboost_reg.predict(self.X_train)
            self.training_data['Train_predictions'] = self.X_y_train_predictions
            # classification report of train data
            print("Xstream Boosting Algorithm Classification report:")
            print(f'train data confusion matrix:\n {confusion_matrix(self.y_train, self.X_y_train_predictions)}')
            print(f'train data accuracy : {accuracy_score(self.y_train, self.X_y_train_predictions)}')
            print(f'train  classification report : {classification_report(self.y_train, self.X_y_train_predictions)}')
            # testing data
            self.X_y_test_predictions = self.Xgboost_reg.predict(self.X_test)
            self.testing_data['Train_predictions'] = self.X_y_test_predictions
            print(f'test data confusion matrix:\n {confusion_matrix(self.y_test, self.X_y_test_predictions)}')
            print(f'test data accuracy : {accuracy_score(self.y_test, self.X_y_test_predictions)}')
            print(f'test  classification report : {classification_report(self.y_test, self.X_y_test_predictions)}')
            # logistic regression
            self.X_reg = LogisticRegression()
            self.X_test_acc_val = cross_val_score(self.X_reg, self.X, self.y, cv=5)
            print(f'cross validation test accuracy : {np.mean(self.X_test_acc_val)}\n')

            # Saving PROJECT
            with open('XB_model.pkl', 'wb') as f:
                pickle.dump(self.Xgboost_reg, f)

        except Exception as e:
            e_t, e_m, e_lin = sys.exc_info()
            print(f'error from line number : {e_lin.tb_lineno} : because {e_t}:')

    def SVC(self):
        try:
            self.svc_reg = SVC(kernel='rbf')
            self.svc_reg.fit(self.X_train, self.y_train)
            self.S_y_train_predictions = self.svc_reg.predict(self.X_train)
            self.training_data['Train_predictions'] = self.S_y_train_predictions
            # classification report of train data
            print("SVC Algorithm Classification report:")
            print(f'train data confusion matrix:\n {confusion_matrix(self.y_train, self.S_y_train_predictions)}')
            print(f'train data accuracy : {accuracy_score(self.y_train, self.S_y_train_predictions)}')
            print(f'train  classification report : {classification_report(self.y_train, self.S_y_train_predictions)}')
            # testing data
            self.S_y_test_predictions = self.svc_reg.predict(self.X_test)
            self.testing_data['Train_predictions'] = self.S_y_test_predictions
            print(f'test data confusion matrix:\n {confusion_matrix(self.y_test, self.S_y_test_predictions)}')
            print(f'test data accuracy : {accuracy_score(self.y_test, self.S_y_test_predictions)}')
            print(f'test  classification report : {classification_report(self.y_test, self.S_y_test_predictions)}')
            # logistic regression
            self.S_reg = LogisticRegression()
            self.S_test_acc_val = cross_val_score(self.S_reg, self.X, self.y, cv=5)
            print(f'cross validation test accuracy : {np.mean(self.S_test_acc_val)}\n')

            # Saving PROJECT
            with open('SVC_model.pkl', 'wb') as f:
                pickle.dump(self.svc_reg, f)

        except Exception as e:
            e_t, e_m, e_lin = sys.exc_info()
            print(f'error from line number : {e_lin.tb_lineno} : because {e_t}:')


if __name__ == '__main__':
    obj = Classification()
    obj.KNN_model()
    obj.Naive_bayes()
    obj.DecisionTree()
    obj.Adaboost()
    obj.Gradientboosting()
    obj.Xstream_boosting()
    obj.SVC()