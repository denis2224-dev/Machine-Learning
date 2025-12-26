#======= 1. import libraries ========
import numpy as np
import pandas as pd

#======= 2. importing the data set =========
dataset = pd.read_csv('Data.csv') 
X = dataset.iloc[:, :-1].values 
#iloc = "integer location" indexing: select by rows / columns positions.
# : - all rows, :-1 all cols but the last one
# .values converts that DataFrame slice into a NumPy array.
# result - 2D array
Y = dataset.iloc[:, -1].values 
#This selects column index 3 (the 4th column) as the target.
#Y becomes a 1D array of labels with shape:

'''
X = inputs / features (what the model learns from)
Y = output / label (what the model tries to predict)
'''

#======= 3. Handling the missing data ===========
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean') #axis = 0 - cols, axis = 1 - rows
imputer = imputer.fit(X[ :, 1:3]) #calculates the mean of each col. Does not change data yet.
X[ :, 1:3] = imputer.transform(X[ :, 1:3]) #replaces every missing value with the learned column mean

#======= 4. Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[ :, 0] = labelencoder_X.fit_transform(X[ :, 0]) #transfor each colummn to numerical values
#Creating a dummy variable 
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [0])
    ],
    remainder='passthrough'
)
X = ct.fit_transform(X)

# Encode Y if categorical
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#======= 5. Splitting the datasets into training sets and Test sets =======
from sklearn.model_selection import train_test_split
'''
Training set → used to learn patterns
Test set → used only to evaluate performance
'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
'''
20% of the data → test set
80% → training set
'''

#======= 6. Feature scaling ========
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() #find formula for standartization
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''
fit: Computes mean and std from X_train only
transform: Applies the scaling using those statistics
'''
print("X shape:", X.shape)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Sample X_train:\n", X_train[:5])
print("Sample Y_train:", Y_train[:5])