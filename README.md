# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![187114741-50011779-d558-46b0-b66e-c290d9fca8c0](https://user-images.githubusercontent.com/94219582/226164790-c96d8074-ba9a-4511-b86a-d69e6b61dc5d.png)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

```
NAME : SITHI HAJARA I
REG NO : 212221230102
        
### To Read CSV file from Google Drive :

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

## To train and test 
from sklearn.model_selection import train_test_split

## To scale 
from sklearn.preprocessing import MinMaxScaler

## To create a neural network model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

### Authenticate User:

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

### Open the Google Sheet and convert into DataFrame :

worksheet = gc.open('sheet_for_DL').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns = rows[0])


df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})

df

X = df[['Input']].values
y = df[['Output']].values

X
y

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size = 0.4, random_state =35)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai_brain = Sequential([
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop' , loss = 'mse')
ai_brain.fit(X_train1 , y_train,epochs = 2005)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()

X_test1 =Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1=[[4]]
X_n1_1=Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```

## Dataset Information

![image](https://user-images.githubusercontent.com/94219582/226168229-6a5244f9-42f2-4592-acfa-78a9e3bbeec0.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/94219582/226168464-67a8912f-4784-42b5-935f-b66be1dd4f05.png)

### Test Data Root Mean Squared Error

![image](https://user-images.githubusercontent.com/94219582/226168511-634fcabb-cad9-4161-bb33-b07d9e7eb623.png)
![image](https://user-images.githubusercontent.com/94219582/226168536-da17fe3c-fa77-402e-af29-716efc23c3ed.png)

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/94219582/226168557-30aca09d-4b93-41f4-9d30-083927da89e9.png)

## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully
