<H3>NAME: Shehan Shajahan </H3>
<H3>REGISTER NO: 212223240154 </H3>
<H3>DATE:  </H3>
<H1>EX.NO.6</H1>
<H1 ALIGN =CENTER>Heart attack prediction using MLP</H1>

<H3>Aim:</H3>  To construct a  Multi-Layer Perceptron to predict heart attack using Python
<H3>Algorithm:</H3>
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<BR>
Step 2:Load the heart disease dataset from a file using pd.read_csv().<BR>
Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<BR>
Step 4:Split the dataset into training and testing sets using train_test_split().<BR>
Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<BR>
Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<BR>
Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<BR>
Step 8:Make predictions on the testing set using mlp.predict(X_test).<BR>
Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<BR>
Step 10:Print the accuracy of the model.<BR>
Step 11:Plot the error convergence during training using plt.plot() and plt.show().<BR>

<H3>Program: </H3>

```python
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('heart.csv')

# Step 2: Separate features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Normalize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Create and train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Step 6: Make predictions on the testing set
y_pred = mlp.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")

# Step 8: Plot the error convergence (loss curve)
plt.plot(mlp.loss_curve_)
plt.title("MLP Training Loss Convergence")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

```

<H3>Output:</H3>

<img width="700" height="500" alt="Screenshot 2025-11-01 120603" src="https://github.com/user-attachments/assets/e5f9dcbf-be0e-4679-b676-6f6142ef611f" />

<H3>Results:</H3>
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
