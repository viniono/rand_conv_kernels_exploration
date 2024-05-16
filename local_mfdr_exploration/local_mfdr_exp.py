import numpy as np
from numba import njit, prange
import pandas as pd
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from scipy.stats import gaussian_kde, norm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial, Gaussian
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegressionCV

## ROCKET FUNCTIONS

# @njit("Tuple((float64[:],int32[:],float64[:],int32[:],int32[:]))(int64,int64)")
@njit("Tuple((float64[:],int32[:],float64[:],int32[:],int32[:]))(int64,int64,int32[:])")
def generate_kernelss(input_length, num_kernels,kernel_len):
    #7,9,11
    # candidate_lengths = np.array((20,20,20), dtype = np.int32)
    candidate_lengths=kernel_len
    lengths = np.random.choice(candidate_lengths, num_kernels)

    weights = np.zeros(lengths.sum(), dtype = np.float64)
    biases = np.zeros(num_kernels, dtype = np.float64)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)

    a1 = 0

    for i in range(num_kernels):

        _length = lengths[i]

        _weights = np.random.normal(0, 1, _length)

        b1 = a1 + _length
        weights[a1:b1] = _weights - _weights.mean()

        biases[i] = np.random.uniform(-1, 1)

        # dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1))) if _length != input_length else 1
        dilation = 2 ** np.random.uniform(0, np.log2(max((input_length - 1) / (_length - 1), 1))) if _length != input_length else 1

        # dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1

    return weights, lengths, biases, dilations, paddings

@njit(fastmath = True)
def apply_kernel(X, weights, length, bias, dilation, padding):

    input_length = len(X)

    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    # _max = np.NINF

    end = (input_length + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):

        _sum = bias

        index = i

        for j in range(length):

            if index > -1 and index < input_length:

                _sum = _sum + weights[j] * X[index]

            index = index + dilation

        # if _sum > _max:
        #     _max = _sum

        if _sum > 0:
            _ppv += 1
# , _max
    return _ppv / output_length

@njit("float64[:,:](float64[:,:],Tuple((float64[::1],int32[:],float64[:],int32[:],int32[:])))", parallel = True, fastmath = True)
def apply_kernels(X, kernels):

    weights, lengths, biases, dilations, paddings = kernels

    num_examples, _ = X.shape
    num_kernels = len(lengths)

    _X = np.zeros((num_examples, num_kernels * 1), dtype = np.float64) # 2 features per kernel

    for i in prange(num_examples):

        a1 = 0 # for weights
        a2 = 0 # for features

        for j in range(num_kernels):

            b1 = a1 + lengths[j]
            b2 = a2 + 2

            _X[i, a2:b2] = \
            apply_kernel(X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])

            a1 = b1
            a2 = b2

    return _X

# Normalize the data
def normalize_data(X):
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True) + 1e-8
    return (X - mean) / std



# Load training data
training_data = np.loadtxt("kernel_size_exploration/UWaveGestureLibraryAll/UWaveGestureLibraryAll_TRAIN.txt")
Y_training, X_training = training_data[:, 0].astype(np.int32), training_data[:, 1:]
test_data = np.loadtxt("kernel_size_exploration/UWaveGestureLibraryAll/UWaveGestureLibraryAll_TEST.txt")
Y_test, X_test = test_data[:, 0].astype(np.int32), test_data[:, 1:]
kernels = generate_kernelss(X_training.shape[-1], 100,np.array((7, 9, 11), dtype=np.int32))
X_training_transform = apply_kernels(X_training, kernels)
X_test_transform = apply_kernels(X_test, kernels)




def local_mfdr(fit, X=None, y=None, method='kernel'):
    """
    Compute local false discovery rates (local fdr) for model coefficients using either
    adaptive shrinkage (ash) or kernel density estimation methods.
    
    Args:
    - fit: Fitted model object from sklearn (LassoCV or LogisticRegressionCV).
    - lambda_val: The penalty strength at which coefficients are evaluated.
    - X (numpy.ndarray): Design matrix.
    - y (numpy.ndarray): Response vector.
    - method (str): Method to compute local fdr ('ash' or 'kernel').
    
    Returns:
    - DataFrame containing estimates, z-scores, and local fdr for penalized variables.
    """
    
    if not hasattr(fit, 'coef_'):
        raise ValueError("The fit must be an sklearn model with a 'coef_' attribute.")
    
    if X is None or y is None:
        raise ValueError("This procedure requires X and y.")
    
    # Extract coefficients at specified lambda value using interpolation or direct extraction
    # Simplification: directly accessing coefficients from fitted model
    # This would ideally be adjusted based on the lambda value with interpolation if necessary.
    beta = fit.coef_
    
    if method not in ['ash', 'kernel']:
        raise ValueError("Method must be 'ash' or 'kernel'.")
    
    # Calculate z-scores and p-values for coefficients
    # Placeholder for actual statistical test computation; simplified for demonstration
    p = beta.shape[0] if len(beta.shape) > 1 else len(beta)
    z = beta / np.std(beta)  # Simplified z-score computation

    if method == 'ash':
        # Placeholder for ASH method; not implemented in Python standard libraries
        # Requires a Python equivalent of the R `ashr` package
        pass
    elif method == 'kernel':
        density = gaussian_kde(z)
        f_z = density(z)
        # The following calculation is a placeholder for local fdr computation using kernel density
        local_fdr = norm.pdf(z) / f_z
        local_fdr = np.minimum(local_fdr, 1)
    
    # Prepare results in DataFrame format
    results = pd.DataFrame({
        'Estimate': beta,
        'z': z,
        'local_fdr': local_fdr
    })
    
    # Select only the penalized coefficients based on some criteria, assuming all here
    penalized_indices = beta != 0
    results = results[penalized_indices]
    
    return results

# This function now needs to be extended to handle multiple types of models
# and the actual computations of z-scores and fdr need to be correctly implemented based on statistical standards.
# Also, proper handling of input arguments, especially lambda values, should be included.


if __name__ == '__main__':
    classifier = LogisticRegressionCV().fit(X_training_transform, Y_training)
    # local_mfdr(classifier,X_training_transform,Y_training)
    