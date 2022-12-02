# linear_regression
Linear regression using numpy

Linear regression is a commonly used statistical method for modeling the relationship between a dependent variable and one or more independent variables. In Python, you can implement linear regression without using the scikit-learn library by manually optimizing the model's parameters using gradient descent.

The numpy library is used to load the data and perform mathematical operations on it. The X and y variables are the data and labels, respectively. A column of ones is added to the data to allow for an intercept term in the linear regression model. The beta variable is used to store the model parameters, which are initialized to zeros. The alpha and n_iter variables are used to specify the learning rate and the number of iterations for gradient descent, respectively.

The for loop is used to iterate over the number of iterations specified by n_iter. At each iteration, the predicted values are computed using the dot() function, and the error is computed as the difference between the predicted values and the true labels. The gradient is then computed using the error and the data, and the model parameters are updated using the gradient and the learning rate.

After the gradient descent loop has finished, the final model parameters are printed. 
