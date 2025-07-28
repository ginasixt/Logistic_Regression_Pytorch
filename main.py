import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import kagglehub
import os

# Download the dataset
path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")

# Compose full CSV path
csv_path = os.path.join(path, "diabetes_binary_health_indicators_BRFSS2015.csv")

# Load data
df = pd.read_csv(csv_path)

# X are features, y is the target variable
X = df.drop(columns=["Diabetes_binary"]).values
y = df["Diabetes_binary"].values

# scaling the features, so that they have mean 0 and variance 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# transform numpy arrays to PyTorch tensors
# unsqueeze is used to convert the target variable to the shape (n_samples, 1) for binary classification
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# input
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim): # input_dim is the number of Input features (21 in this case)
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1) # creating a linear layer with input_dim features and 1 output (for binary classification)

    # self.linear() is the linear layer (nn.Linear) that calculates a weighted sum from input data, the weighting is optimized during training by gradient descent.
    # then we run the result through a sigmoid activation function to get probabilities between 0 and 1.
    def forward(self, x):
        return torch.sigmoid(self.linear(x)) #self.linear(x) computes the linear transformation, and 
    

# shape[0] gives the number of samples, shape[1] gives the number of features, becausse X_train ist ein 2D-Tensor in the Form (samples, features)
# now we can create an instance of our model with the number of input features
model = LogisticRegressionModel(X_train.shape[1]) 

# Create loss function and optimizer
# Creating a SGD-Optimizer (Stochastic Gradient Descent) to adjust the weights of the model during training.
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# In jeder Iteration:
# 1) Vorhersagen
# 2) Loss berechnen
# 3)Alte Gradienten löschen
# 4) Neue Gradienten berechnen
# 5) Gewichte anpassen
# Training for 1000 epochs/ iterations of whole dataset
for epoch in range(1000):
    # Set the model to training mode
    model.train()
    # Forward pass (forward(self, x)): compute predicted outputs by passing inputs to the model
    outputs = model(X_train) 
     # Calculate the loss using Binary Cross Entropy Loss (BCELoss) between predicted outputs and true labels.
    loss = criterion(outputs, y_train)
     # .backward() sums the gradients, so we need to zero them out before the next iteration
    optimizer.zero_grad()
    # the gradient is the Richtungszeicher, wie das Gewicht verändert werden soll. (wie Gewicht verändert werden soll, dass Loss geniger wird)
    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward() 
    # Update model parameters based on gradients (Paramter Update) with the Gradient Descent Update Rule
    optimizer.step() 
    
    # Print loss every 100 iterations
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# testing the model with the test data
# In evaluation mode, the model will not update its weights and will not apply dropout or batch
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test)
    y_pred = (y_pred_probs >= 0.5).float()


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
