# Diabetes Prediction with Logistic Regression with PyTorch

This project uses the "Diabetes Health Indicators" dataset from Kaggle to train and evaluate a logistic regression model that predicts whether a person has diabetes based on health-related features (e.g., BMI, age, blood pressure, etc.). \
To better understand the inner workings of logistic regression and prepare for neural network projects, a second version of this project was built using PyTorch.
##  Key Results

```text
Epoch 0, Loss: 0.7178
Epoch 100, Loss: 0.5688
Epoch 200, Loss: 0.4916
Epoch 300, Loss: 0.4450
Epoch 400, Loss: 0.4146
Epoch 500, Loss: 0.3936
Epoch 600, Loss: 0.3787
Epoch 700, Loss: 0.3677
Epoch 800, Loss: 0.3594
Epoch 900, Loss: 0.3530
```
Despite using a different framework, the PyTorch model shows very similar performance to the sklearn version, \
confirming consistency in logic and data handling: 

```text
Accuracy: 0.8648691264585304
Confusion Matrix:
 [[42887   852]
 [ 6004   993]]
Classification Report:
               precision    recall  f1-score   support

         0.0       0.88      0.98      0.93     43739
         1.0       0.54      0.14      0.22      6997

    accuracy                           0.86     50736
   macro avg       0.71      0.56      0.58     50736
weighted avg       0.83      0.86      0.83     50736
```
