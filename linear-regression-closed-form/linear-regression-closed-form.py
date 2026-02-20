import numpy as np

def linear_regression_closed_form(X, y):
    X=np.array(X)
    y=np.array(y)
    x_transpose= X.T
    a=x_transpose @ X
    inverse_a= np.linalg.inv(a)
    theta=inverse_a @ x_transpose @ y
    

    return theta.flatten()

