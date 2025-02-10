import numpy as np

def gradientdescent(x,y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.08

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        print('m {}, b {}, cost {}, iterations {}'.format(m_curr,b_curr,cost, i))


x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradientdescent(x,y)






# ABOUT THE CONCEPT OF GRADIENT DESCENT
#
# Gradient Descent is an optimization algorithm used to find the minimum value of a function.
# In machine learning, it is commonly used to minimize the cost function in linear regression and
# other models.
#
# Key Concepts:
#
# 1. **Cost Function**: This measures how well the model's predictions match the actual data.
# In this case, the cost function is the Mean Squared Error (MSE), which calculates the average
# squared difference between predicted and actual values.
#
# 2. **Learning Rate**: This is a hyperparameter that controls the size of the steps taken
# towards minimizing the cost function. A larger learning rate might lead to faster convergence
# but can overshoot the minimum, while a smaller learning rate ensures more precise convergence but can be slower.
#
# 3. **Gradient**: The gradient is the derivative of the cost function with respect to the
# model parameters (slope `m` and intercept `b`). It indicates the direction and rate of the
# steepest ascent. To minimize the cost function, we move in the opposite direction of the gradient.
#
# 4. **Iterations**: The number of times the algorithm updates the model parameters. More
# iterations allow for more refinement of the parameters but also require more computation.
#
# **Algorithm Steps**:
# 1. Initialize parameters `m` (slope) and `b` (intercept) to zero.
# 2. Compute predictions using the current parameters.
# 3. Calculate the cost (error) of the predictions.
# 4. Compute the gradients (derivatives) for `m` and `b`.
# 5. Update the parameters `m` and `b` using the gradients and learning rate.
# 6. Repeat steps 2-5 for a specified number of iterations or until convergence.
#
# In this code, `gradientdescent` function iteratively updates the parameters
# `m` and `b` to minimize the cost function, and prints the values of parameters
# and cost at each iteration.
