<H3>Name : AUGUSTINE J</H3>
<H3>REGISTER NO: 212222240015</H3>
<H3>EX. NO.1</H3>

<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>


## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
### IMPORT LIBRARIES :
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```
### READ THE DATA:
```
df=pd.read_csv("Churn_Modelling.csv")
```
### CHECK DATA:
```
df.head()
df.tail()
df.columns
```
### CHECK THE MISSING DATA:
```
df.isnull().sum()
```
### ASSIGNING X:
```
X = df.iloc[:,:-1].values
X
```
### ASSIGNING Y:
```
Y = df.iloc[:,-1].values
Y
```
### CHECK FOR OUTLIERS:
```
df.describe()
```
### DROPPING STRING VALUES DATA FROM DATASET:
```
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
```
### CHECKING DATASETS AFTER DROPPING STRING VALUES DATA FROM DATASET:
```
data.head()
```
### NORMALIE THE DATASET USING (MinMax Scaler):
```
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```
### SPLIT THE DATASET:
```
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
print(X)
print(Y)
```
### TRAINING AND TESTING MODEL:
```
X_train ,X_test ,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```
## OUTPUT:
### DATA CHECKING:
![image](https://github.com/user-attachments/assets/bfb7d92a-7282-460d-8c73-01ac8f3365d3)
### MISSING DATA:
![image](https://github.com/user-attachments/assets/f891b9df-2136-49f7-8649-5a5d807cdd4c)
### DUPLICATES IDENTIFICATION:
![361235546-112b193c-5bb8-425a-92a7-8f30f5903949](https://github.com/user-attachments/assets/b6e7f215-b39a-49ad-adf0-ea34e532111b)
### VALUE OF Y:
![image](https://github.com/user-attachments/assets/dfa01088-19e3-4d80-8cd4-85bcfd729772)

### OUTLIERS:
![image](https://github.com/user-attachments/assets/a0ca4d9b-6461-496d-bc9b-71de409c4dc3)
### CHECKING DATASET AFTER DROPPING STRING VALUES DATA FROM DATASET:
![image](https://github.com/user-attachments/assets/32cc5e85-f225-4654-989c-9d987b6d8b73)
### NORMALIZE THE DATASET:
![image](https://github.com/user-attachments/assets/825c6ecf-2cb7-4a8b-85db-b52eabdc3f61)
### SPLIT THE DATASET:
![image](https://github.com/user-attachments/assets/d6446974-d58c-470e-a7de-8885a930f3bf)
### TRAINING AND TESTING MODEL:
![image](https://github.com/user-attachments/assets/2e79489c-e52f-4de8-9a33-bc7dc3ed2a37)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


