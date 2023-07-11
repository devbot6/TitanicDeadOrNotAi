import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#This is the rawdata csv file
raw_data = pd.read_csv('C:/Users/devhu/Desktop/AllCodeStuff/MLBootcamp/data.csv')
raw_data = raw_data.dropna()
raw_data.head()


#will tell you what type of object the variable raw_data is. Running the code,
#you will see that raw_data is a Dataframe object.
#Dataframes are a custom object provided by pandas that is very useful for data analysis. 
type(raw_data)

#This drops all the rows of data that we don't want
raw_data.drop(["PassengerId", "Cabin", "Embarked",
               "Name", "SibSp", "Parch", "Ticket"], axis = 1, inplace = True)

raw_data.head()

#This assigns numbers as labels (0 or 1) instead of the string label currently in the data. 
raw_data['Sex'] = np.where(raw_data['Sex'] == 'male', 0, raw_data['Sex'])# read this like "where raw_data['Sex'] = male
#replace 'male' with 0. Otherwise, just use the value in raw_data['Sex']. 
raw_data['Sex'] = np.where(raw_data['Sex'] == 'female', 1, raw_data['Sex'])

#This code uses the library scikit-learn (which we talked about earlier) to split our raw data into train, test, and validation.
train_data ,test_and_val_data = train_test_split(raw_data, test_size = .20, random_state = 2) #split into train and an aggragated test and val
val_data, test_data = train_test_split(test_and_val_data, test_size = .50, random_state = 2)#split aggregagted into test and val


#split the test data into input and output. "Y" is our labels for who survived. "X" are our input variables: Purchasing class and sex. 
#Then, we make a Random Forest Model with 10 trees in our forest, and a max_depth of 5 on each tree. 
#This section splits the data, this is the data that 
Y = train_data['Survived']
X = train_data[['Pclass', 'Sex', 'Fare', 'Age']]
model = RandomForestClassifier(10, max_depth = 5)
#This is where we fit the data which is when the model actually does the learning we are fitting the algorithm into our training data
model.fit(X,Y)


#This is the validation data that ensures that the model works
val_Y = val_data['Survived']
val_X = val_data[['Pclass', 'Sex', 'Fare', 'Age']]
predictions = model.predict(val_X)

#This caluates and prints out the accuracy of the model
accuracy = np.sum(val_Y == predictions)/len(val_X)*100
print("Accuracy: " , accuracy)