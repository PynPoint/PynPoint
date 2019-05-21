"""
Module for statistical resampling
"""
import numpy as np

def jackknife_estimator(data, n, estimator, axis=0):
    """
    Grabs n samples out of the data and calculates a jackknife estimator on top along the 0th axis.
    Source: Statistical Methods in Astrophysics and Cosmology Lecture ETHZ HS2019

    Parameters
    ----------

    data : numpy.array
        data on which the estimator should be applied
    n : int
        Size of the subsample on which the *method* estimator is calculated.\
        Must be smaller or equal than data.shape[axis]
    estimator : method
        Function which calculates the estimated value

    Returns
    -------
    est : numpy.array
        Estimated numpy.array with one less dimension along the selected axis compared to data
    variance : numpy.array
        Variance of the jackknife estimator
    """

    #choose n values at random from the underlying distribution
    indices = np.random.choice(a=data.shape[0], size=n, replace=False)
    data = data[indices]

    theta = estimator(data, axis=0)
    theta_i = np.zeros((n, *data.shape[1:]))

    for i in range(n):
        # remove ith entry for indices
        no_i_indices = list(indices)
        del no_i_indices[i]
        theta_i[i] = np.sum(estimator(data[no_i_indices], axis=0), axis=0)

    theta_i /= n-1
    theta_dot = np.sum(theta_i, axis=0)

    bias = (n-1) * (theta_dot - theta)
    variance = (n-1) / n * np.sum([(theta_i[i] - theta_dot)**2 for i in range(n)], axis=0)
    return theta - bias, variance

def bootstrap_estimator(data, n, bootstrap_samples, estimator):
    """
    Grabs n samples out of the data and calculates a bootstrap estimator on top along the 0th axis.
    Source: Statistical Methods in Astrophysics and Cosmology Lecture ETHZ HS2019

    Parameters
    ----------

    data : numpy.array
        data on which the estimator should be applied
    n : int
        Size of the subsample on which the *method* estimator is calculated.\
        Must be smaller or equal than data.shape[axis]
    estimator : method
        Function which calculates the estimated value

    Returns
    -------
    est : numpy.array
        Estimated numpy.array with one less dimension along the selected axis compared to data
    variance : numpy.array
        Variance of the jackknife estimator
    """
    #choose n values at random from the underlying distribution
    indices = np.random.choice(a=data.shape[0], size=n, replace=False)
    theta = estimator(data, axis=0)

    bootstrap_indices = np.random.choice(indices, size=(bootstrap_samples, n), replace=True)

    theta_i = np.zeros((n, *data.shape[1:]))

    for i, bootstrap_sample in enumerate(bootstrap_indices):
        theta_i[i] = np.sum(estimator(data[bootstrap_sample], axis=0), axis=0)

    theta_i /= bootstrap_samples
    theta_dot = np.sum(theta_i, axis=0)

    bias = theta_dot - theta
    variance = 1 / n * np.sum([(theta_i[i] - theta_dot)**2 for i in range(n)], axis=0)
    return theta - bias, variance
