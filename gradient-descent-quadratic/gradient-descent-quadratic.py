def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    x=x0
    for i in range(steps):
        dx=(2*a*x)+b
        x=x-(lr * dx)
    return x
    pass