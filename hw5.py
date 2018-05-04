# -*- coding: utf-8 -*-
"""
Created on Wed May  2 18:01:25 2018

@author: shahharsh85
"""
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
warnings.filterwarnings('ignore')

def Preprocessing(df,target_col):
    print("Starting preprocessing----------")
    print("Removing rows where target column is null")
    df = df[np.isfinite(df[target_col])]
    print("Handling outliers")
    df.drop(df.index[[885,992,676]],axis=0,inplace=True)
    #df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
    df2 = pd.get_dummies(df, columns=['Smoking', 'Alcohol','Punctuality', 'Lying','Internet usage','Gender', 'Left - right handed', 'Education', 'Only child', 'Village - town', 'House - block of flats'])
    print("Working with categorical data")
    #df2=df.drop(['Smoking', 'Alcohol','Punctuality', 'Lying','Internet usage','Gender', 'Left - right handed', 'Education', 'Only child', 'Village - town', 'House - block of flats'],axis=1)
    print("Filling the missing values")
    df2.fillna(df2.median(),inplace=True)
    return df2

    
def CreateSplits(X,y):
    #splitting the data into train development
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=101)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=101)
    return X_train,y_train,X_val,y_val,X_test,y_test
    

def over_sample(train_vectors, train_class):
    #oversampling to make equal number of target class distribution
    sm = SMOTE(random_state=42)
    train_vectors, train_class = sm.fit_sample(train_vectors, train_class)
    return train_vectors, train_class        


def decision_tree(X_train,y_train,X_val,y_val,X_test,y_test):
    max_i=0
    max_acc=0
    max_j=0
    _,col=X_train.shape
    #oversample input
    X_train, y_train = over_sample(X_train, y_train)
    for i in np.arange(20,30):
        for j in np.arange(1,40):
            #tuning hyperparameters
            dtree = DecisionTreeClassifier(max_features=i,max_depth=j)
            dtree.fit(X_train,y_train)
            #run on validation
            predictions = dtree.predict(X_val)
            if (max_acc<np.mean(y_val==predictions)):
                max_i=i
                max_j=j
                max_acc=np.mean(y_val==predictions)
    dtree = DecisionTreeClassifier(max_features=max_i,max_depth=max_j)
    dtree.fit(X_train,y_train)
    #run on test
    predictions = dtree.predict(X_test)
    print("Accuracy of Decision Tree:")
    print(np.mean(y_test==predictions))
    print("")
    
def KNN(X_train,y_train,X_val,y_val,X_test,y_test):
    max_i=0
    max_acc=0
    #tuning hyperparameter K
    for i in range(7,41):
        knn = KNeighborsClassifier(n_neighbors=i,weights='distance')
        knn.fit(X_train,y_train)
        pred_i = knn.predict(X_val)
        #run on validation
        if (max_acc<np.mean(y_val==pred_i)):
                max_i=i
                max_acc=np.mean(y_val==pred_i)
    knn = KNeighborsClassifier(n_neighbors=max_i,weights='distance')
    #best parameter on test set
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    print("Accuracy of KNN :")
    print(np.mean(y_test==pred_i))
    print("")
    
def MLP(X_train,y_train,X_val,y_val,X_test,y_test):
    #tuning hyper parameters using grid search
    gs = GridSearchCV(MLPClassifier(), param_grid={
    'solver' : ['lbfgs', 'sgd', 'adam'],
    'alpha': [0.00001,0.0001,0.001,0.01,0.01],
    'hidden_layer_sizes':[(15, 25)],
    'random_state': [1]
            })
    gs.fit(X_train, y_train)
    grid_predictions = gs.predict(X_val)
    print("Accuracy for MLP:")
    print(np.mean(y_val==grid_predictions))
    print("")
   
    
def my_model(X,y):
    print("Runnig my_model-------")
    rows,col=X.shape
    # models in the classifier
    clf1 = ExtraTreesClassifier(n_estimators=80, max_depth=None,min_samples_split=2)
    clf2 = RandomForestClassifier(n_estimators=250, max_depth=None,random_state=101)
    clf3 = AdaBoostClassifier(n_estimators=80)
    clf4 = GradientBoostingClassifier(n_estimators=180, learning_rate=1.0,max_depth=1)
    clf5 = RandomForestClassifier(random_state=1)
    clf8 = ExtraTreesClassifier(n_estimators=80, max_depth=None,min_samples_split=2)
    clf6 = GaussianNB()
    clf7 = SVC(kernel='rbf', probability=True)
    clf10 = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=7)
    clf8 = SVC(kernel='poly',probability=True)
    clf9 = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

    max_acc=0
    max_i=0
    print("Training model and tuning hyperparameters")
    for i in np.arange(64,67):
        X_new = SelectKBest( k=i).fit_transform(X, y)
        X_train,y_train,X_val,y_val,X_test,y_test=CreateSplits(X_new,y)
        eclf = VotingClassifier(estimators=[('et', clf1), ('rfc', clf2), ('ab', clf3),('gb', clf4),('rfc1', clf5),('gnb', clf6),('svc',clf7),('svc1',clf8),('mlp',clf9),('bc',clf10)], voting='soft')
        eclf = eclf.fit(X_train,y_train)
        pred=eclf.predict(X_val)
           
        if(max_acc<np.mean(y_val==pred)):
            max_acc=np.mean(y_val==pred)
            max_i=i
    print("Best features selected now check test set")    
    X_new = SelectKBest( k=max_i).fit_transform(X, y)
    X_train,y_train,X_val,y_val,X_test,y_test=CreateSplits(X_new,y)
    eclf = VotingClassifier(estimators=[('et', clf1), ('rfc', clf2), ('ab', clf3),('ab1', clf3),('gb', clf4),('rfc1', clf5),('gnb', clf6),('svc',clf7),('svc1',clf8),('svc3',clf8),('mlp',clf9),('bc',clf10)], voting='soft')
    eclf = eclf.fit(X_train,y_train)
    pred=eclf.predict(X_test)
    print("Accuracy for my_model")
    print(np.mean(pred==y_test))
        

        
        
def run_models(X_train,y_train,X_val,y_val,X_test,y_test):    
    print("Running Decision tree")
    decision_tree(X_train,y_train,X_val,y_val,X_test,y_test)
    print("Running KNN tree")
    KNN(X_train,y_train,X_val,y_val,X_test,y_test)
    print("Running MLP tree")
    MLP(X_train,y_train,X_val,y_val,X_test,y_test)

target='Spending on healthy eating'
df= pd.read_csv('responses.csv')
df=Preprocessing(df,target)
X=df.drop(target,axis=1)
y=df[target]
print("Splitting data into train/dev/test-----------\n")
X_train,y_train,X_val,y_val,X_test,y_test=CreateSplits(X,y)
print("running base models \n")
run_models(X_train,y_train,X_val,y_val,X_test,y_test)

my_model(X,y)
