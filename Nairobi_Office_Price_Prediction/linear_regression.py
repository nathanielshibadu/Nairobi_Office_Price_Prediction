import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Nairobi Office Price Ex.csv')  

# Isolate the SIZE (feature) and PRICE (target) columns
X = data['SIZE'].values
y = data['PRICE'].values

# Define Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Define Gradient Descent function for linear regression
def gradient_descent(X, y, m, c, learning_rate, epochs):
    n = len(y)  # Number of data points
    mse_history = []  # To store MSE for each epoch
    
    for epoch in range(epochs):
        # Predictions based on current m and c
        y_pred = m * X + c
        
        # Compute current MSE
        mse = mean_squared_error(y, y_pred)
        mse_history.append(mse)
        
        # Calculate gradients
        m_gradient = -(2/n) * np.sum(X * (y - y_pred))
        c_gradient = -(2/n) * np.sum(y - y_pred)
        
        # Update parameters
        m -= learning_rate * m_gradient
        c -= learning_rate * c_gradient
        
        # Print MSE for the current epoch
        print(f"Epoch {epoch+1}, MSE: {mse}")
        
    return m, c, mse_history

# Initialize parameters
m_initial = np.random.rand()  # Random initial slope
c_initial = np.random.rand()  # Random initial y-intercept
learning_rate = 0.01
epochs = 10

# Train the model using Gradient Descent
m_trained, c_trained, mse_history = gradient_descent(X, y, m_initial, c_initial, learning_rate, epochs)

# Plot the line of best fit after final epoch
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, m_trained * X + c_trained, color='red', label='Best fit line')
plt.xlabel("Office Size (sq. ft.)")
plt.ylabel("Office Price")
plt.title("Line of Best Fit After Training")
plt.legend()
plt.show()

# Predict the office price for a size of 100 sq. ft.
predicted_price = m_trained * 100 + c_trained
print("Predicted price for 100 sq. ft.:", predicted_price)
