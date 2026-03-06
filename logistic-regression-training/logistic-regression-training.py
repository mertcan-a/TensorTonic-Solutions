import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X=np.array(X)
    y=np.array(y)
    w=np.zeros(X.shape[1])
    b=0.0

    for i in range(steps):
        
        z=(X @ w) + b
        prediction=_sigmoid(z)

        dz=prediction-y
        dw=(1/X.shape[0]) * X.T @ dz
        db=(1/X.shape[0]) * np.sum(dz)


        w= w-(lr * dw)
        b= b-(lr * db)
    return w , b
    pass