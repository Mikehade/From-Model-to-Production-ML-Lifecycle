#importing needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from sklearn.manifold import LocallyLinearEmbedding
from sklearn import manifold, datasets

from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler
from sklearn import preprocessing

from sklearn.feature_selection\
    import VarianceThreshold
from sklearn.feature_selection import VarianceThreshold

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, plot_roc_curve, confusion_matrix

import mlflow
import mlflow.sklearn

from os.path import exists

from flask import Flask, flash, redirect, render_template, request
import csv
from multiprocessing import Process


from sklearn.model_selection import train_test_split
#from tests import feat_sel_col_test

import calendar
#import datetime
from datetime import date
from datetime import time
from datetime import datetime

import warnings


def main():
    warnings.filterwarnings("ignore")

    #Check files status to know if static and continuous files exist
    file_status("creditcard_static.csv", "creditcard_cont.csv")

    #create static dataframe by opening static dataset
    df_static = wrangle("creditcard_static.csv")

    #create cotinuous dataframe by opening continuous dataset
    df_cont = wrangle("creditcard_cont.csv")



    #print(new_df_static.shape)
    flask('creditcard_cont.csv')
    max_inp = int(index)

    for i in range(max_inp):

        #print(new_df_cont.loc[[i]])
        #new_data =

        df_static = df_static.append(df_cont.loc[[i]])
        print(df_static.shape)

        new_df_static = feat_sel(df_static)
        print(new_df_static.shape)
        #split static dataset into test and train
        x_train, x_test, y_train, y_test = test_train_split(new_df_static)

            #perform Logistic Regression
        model = modelling(x_train, y_train)    #x_train, x_test, y_train, y_test

            #loging with mlflow
        loggings(x_train, y_train, x_test, y_test, model)


            #end loggings
        end_log()
    
    #monthly schedule of learning
    monthly_schedule = schedule()
    if monthly_schedule == True:
        main()







def file_status(datapath1, datapath2):
    first_file_exists = exists(datapath1)
    sec_file_exists = exists(datapath2)

    if first_file_exists == True and sec_file_exists == True:
        print("Main File Exists")

    else:

        df = pd.read_csv("creditcard.csv")
        #create static dataset and save to csv
        df_static = df[:(int(len(df) * 0.95))]

        #crate dataset that will be iterated to mimic continuous dataset
        df_cont = df[(int(len(df) * 0.95)):]
        df_static.to_csv("creditcard_static.csv", index=False)
        df_cont.to_csv("creditcard_cont.csv", index=False)
        #return df_static, df_cont


def wrangle(datapath):
    #read data
    df = pd.read_csv(datapath)

    return df

def feat_sel(data):
    """Feature Variance to reduce features
    by select important features with highest variance """

    #X = data.iloc[:,0:30]    #select first thirty features for selection of important ones
    #y = data.iloc[:,30]   #isolating output feature

    #isolating features and labels(output variables)
    leng = len(data.columns) - 1
    X = data.iloc[:,0:leng]
    y = data.iloc[:,leng]


    selector = VarianceThreshold(threshold=0.90) #using a threshold of 90 percent
    Var = selector.fit_transform(X)

    X = data[data.columns[selector.get_support(indices=True)]]

    z = pd.concat([X, y], axis=1)  #concatenate features and labels after features has been reduced
    return z


def test_train_split(data):

    #isolating features and labels
    leng = len(data.columns) - 1
    X = data.iloc[:,0:leng]
    y = data.iloc[:,leng]

    #split into train and test
    x_train, x_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2, random_state = 2020)

    #scaling
    scaler = StandardScaler()

    x_train = pd.DataFrame(scaler.fit_transform(x_train))
    x_test = pd.DataFrame(scaler.fit_transform(x_test))

    return x_train, x_test, y_train, y_test


def modelling(x_train, y_train):  #x_train, xtest, y_train, y_test

    model = LogisticRegression(random_state=None, max_iter=400, solver='newton-cg')
    model.fit(x_train, y_train)

    return model

def evaluation(model, x_test, y_test):

    eval_ = (model.score(x_test, y_test) * 100)  #use learned model to compute evaluation score
    preds = model.predict(x_test)   #use learned model to get predictions  (model.predict(x_test) * 100)
    auc = (roc_auc_score(y_test, preds) * 100)
    #log evaluation score and auc as metrics
    mlflow.log_metric("Evaluation Score", eval_)
    mlflow.log_metric("Area under Curve",auc)

    #callling of confusiion matrix function
    plot_mat(y_test, preds)
    mlflow.log_artifact("Model_Confusion_Matrix.png")

    #calling of roc curve function
    plot_roc(x_test, y_test, model)
    mlflow.log_artifact("Model_ROC_Curve.png")



    return auc, eval_, preds


def plot_roc(x_test, y_test, model):

    #roc curve
    roc_plot = plot_roc_curve(model, x_test, y_test, name='Model ROC Curve')
    plt.savefig("Model_ROC_Curve.png")


def plot_mat(y_test, y_pred):

    #confusion matrix
    mat = confusion_matrix(y_test, y_pred)
    axi = sns.heatmap(mat, annot=True, fmt='g')
    axi.invert_xaxis()
    axi.invert_yaxis()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("Model_Confusion_Matrix.png")



def loggings(x_train, y_train, x_test, y_test, model):
    sci_model = LogisticRegression(random_state=None, max_iter=400, solver='newton-cg')
    mlflow.set_experiment("Fraud Detection")
    with mlflow.start_run():
        #train(sci_model, x_train, y_train)
        modelling(x_train, y_train)
        #evaluate(sci_model, x_test, y_test)
        evaluation(model, x_test, y_test)
        mlflow.sklearn.log_model(sci_model, "log_reg_model")
        print("Model run: ", mlflow.active_run().info.run_uuid)
    #mlflow.end_run()

def end_log():
    mlflow.end_run()


def flask(datapath):
    #warnings.filterwarnings("ignore")

    lists = []
    save = []
    app = Flask(__name__)
    with open(datapath, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            lists.append(row)
    #global a
    a = "on"

    @app.route("/", methods=["GET", "POST"])
    def index():
        global index
        message = "Please enter an Index value"
        if request.method == "GET":
            return render_template("index.html", list=message)  #list=lists[0]

        else:
            index = request.form.get("index")
            try:
                int(index)
            except:
                return render_template("index.html", list=message)


            if not index:
                return render_template("index.html", list=message)
            #elif int(index) > len(lists):
                #return render_template("index.html", list=lists[0])
            else:
                save = lists[int(index)]
                #print(save)
                return render_template("index.html", list=f"Learning with first {index} records in continuous dataset") #list=lists[int(index)]

    if a == "on":
        print("server running")
        server = Process(target=app.run(port=2005))
        server.start()
    
    @app.route("/shutdown", methods=["POST"])
    def shutdown():
        #server.terminate()
        a = "off"
        if a == "off":
            server.terminate()
        print(a)
        return redirect("/")


    #server = Process(target=app.run(port=2005))
    

    return index

def schedule():

    t = datetime.now()
    year = t.year
    day = t.day     #stores day
    mon = t.month
    ho = t.hour
    minut = t.minute

    last_day_mon = (calendar.monthrange(year, mon)[1])

    
    if day == last_day_mon and ho == 9 and minut == 30:  #checks if date is last day of the month at 9:30am
        return True
    else:
        return False





if __name__ == '__main__':
    main()
