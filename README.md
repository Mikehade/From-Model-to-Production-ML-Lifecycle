# From-Model-to-Production-ML-Lifecycle
Automating and Tracking Supervised Machine Learning program using MLFlow. Use of Unsupervised machine learning for Fraud detection using Logistic Regression for classification performed on a dataset in a survey downloaded from this [link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) to Kaggle website .
## Below is the main process of managing the model to production lifecycle.
## Main steps: -
- Data processing : - Processing of data by loading to dataframe and splittiong to static and continuous datasets. The continuous dataset will be loaded iteratively to   the machine learning algorithm through Flask API to simulate continuous data for continuous learning in deployment.
- Feature Selection: - selecting of important features and getting rid of irrelevant features using feature variance.
- Modelling: - Using Logistic regression to classify fraudulent and non-fradulent transactions.
- Evaluation: - Using metrics like confusion matrix and ROC curve to monitor model performance.
- Automation and Tracking: - using Mlflow to automate and keep track of machine learning lifecycle.
### Instructions on usage
- Download dataset from link and copy into same folder containing templete, fraud.py files
- run the python file from command prompt in the project directory
- open the flask link from command prompt in browser
- enter amount of records into the **index** box and click enter
- then click shutdown on webpage
- On the command prompt click **Ctrl + c** to shut down flask completely and learning will continue
- You can enter the following command **mlflow ui -p 1234** into a seperate command prompt but same directory as before.
- On a different webpage insert the following [link](http://localhost:1234/), this is the page which you will find all logged metrics and artifacts.
