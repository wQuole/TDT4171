__author__ =  "wQuole"
import numpy as np


# Transition matrix
T = np.array([
    [0.7, 0.3],
    [0.3, 0.7]
    ])

# Observation matrix
O = np.array([
    [[0.1, 0.0],
     [0.0, 0.8]],   # without umbrella

    [[0.9, 0.0],
     [0.0, 0.2]]    # with umbrella
])


# Normalize
def normalize(v, desired_sum=1):
    '''
    Normalizes v
    :param v:   numpy.ndarray # A vector
    :return:    numpy.ndarray # Normalized vector
    '''
    return v*(desired_sum/np.sum(v))


# Compute P(Z_k, x_1:x_k)
def forward(fv, ev):
    """
    Equation 15.12 in the book
    :param fv:  numpy.ndarray    # forward vector
    :param ev:  numpy.ndarray    # evidence vector
    :return:    numpy.ndarray
    """
    # return normalize(O[ev].dot(T.T.dot(fv)))   # if 1-D arrays
    return normalize(O[ev] @ T @ fv)            # 2-D arrays


# Compute P(x_(k+1):x_n | x_1:x_k)
def backward(bv, ev):
    """
    Equation 15.13 in the book
    :param bv:  numpy.ndarray    # backward vector
    :param ev:  numpy.ndarray    # evidence vector
    :return:    numpy.ndarray
    """
    #return T.dot(O[ev].dot(bv))    # 1-D arrays
    return (T @ O[ev] @ bv)         # 2-D Arrays



# Compute P(Z_k | x_1:x_n)
def fb(prior, ev):
    '''
    Forward-Backward Algorithm
    Assumed known:
      P(X_k | Z_k)
      P(Z_k | Z_(k+1))
      P(Z_1)
    :param ev:      numpy.ndarray   # evidence vector
    :param prior:   numpy.ndarray # prior probability = [0.5, 0.5]
    :return:        numpy.ndarray
    '''
    fv = [prior] #forward vec
    bv = np.array([1, 1]) # initialize backward vec
    sv = [] # smoothing vec
    N = len(ev) # amount of evidence

    fv[0] = prior
    for i in range(1,N+1):
        fv.append(forward(fv[i-1], ev[i-1]))

    print("Backward messages:")
    for i in range(N-1, -1, -1):
        sv.append(normalize(fv[i+1] * bv))
        bv = backward(bv, ev[i])
        print("b:{} --> {}".format(i+1, bv))
    return sv


# TASK B
def run_forward(fv, ev):
    """
    Testrun the forward implementation
    :param fv:  numpy.ndarray    # forward vector
    :param ev:  numpy.ndarray    # evidence vector
    :return:    None
    """
    f_copy = fv.copy() # Make a copy to avoid augmenting the original
    N = len(ev)
    print("The probability of rain over a period of {} day(s)".format(N))
    for i in range(N):
        f_copy = forward(f_copy, ev[i])
        print("Day {}: {}".format(i+1, f_copy))


# TASK C
def run_forward_backward(fv, ev):
    """
    Testrun the forward-backward implementation
    :param fv:  numpy.ndarray    # forward vector
    :param ev:  numpy.ndarray    # evidence vector
    :return:    None
    """
    sv = fb(fv, ev)
    N = len(ev)
    print("\nThe probability of rain over a period of {} day(s)".format(N))
    for i in range(N):
        print("Day {}: {}".format(i+1, sv[i]))



def main():
    # Evidence
    B = np.array([1, 1, 0, 1, 1])  # 1 = True, 0 = False

    # Probability for rain at init
    PRIOR = np.array([0.5, 0.5])  # Assumes equal prob for rain or no.

    print(" * * * PART B * * * ")
    print("2 days...")
    run_forward(PRIOR, B[0:2])
    print("\n5 days...")
    run_forward(PRIOR, B)

    print("\n * * * PART C * * * ")
    print("2 days...")
    run_forward_backward(PRIOR, B[0:2])
    print("\n5 days...")
    run_forward_backward(PRIOR, B)


if __name__ == "__main__":
    main()

