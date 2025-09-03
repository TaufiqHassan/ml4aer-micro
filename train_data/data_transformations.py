import numpy as np

# Transformation functions
def standard_transform(stats, data, key_mean, key_std):
    """Standardize data using mean and std from stats dictionary."""
    return (data - stats[key_mean]) / stats[key_std]

def inverse_standard_transform(stats, data_transformed, key_mean, key_std):
    """Revert standardized data to original scale."""
    return data_transformed * stats[key_std] + stats[key_mean]

def log_transform(x):
    """Apply log transformation with small epsilon to avoid issues with zero values."""
    return np.log(np.abs(x) + 1e-8)

def exp_transform(x):
    """Apply exponential transformation."""
    return np.exp(x) - 1e-8

def combined_log_standard_transform(data, stats, y=False):
    """Apply log and standardization transformations combined."""
    log_data = log_transform(data)
    if y:
        return standard_transform(stats, log_data, 'ytrain_log_mean', 'ytrain_log_std')
    return standard_transform(stats, log_data, 'log_mean', 'log_std')

def inverse_combined_log_standard_transform(data_transformed, stats, y=False):
    """Revert combined log and standardization transformations."""
    if y:
        standard_data = inverse_standard_transform(stats, data_transformed, 'ytrain_log_mean', 'ytrain_log_std')
    else:
        standard_data = inverse_standard_transform(stats, data_transformed, 'log_mean', 'log_std')
    return exp_transform(standard_data)

def apply_transformation(Xtrain, yTrain, Xval, yval, Xtest, ytest, transformation_type='standardization'):
    # Calculate mean and std for standardization
    xtrain_mean = np.mean(Xtrain, axis=0)
    xtrain_std = np.std(Xtrain, axis=0)
    ytrain_mean = np.mean(yTrain, axis=0)
    ytrain_std = np.std(yTrain, axis=0)
    
    # For log-standardization, calculate the mean and std of the log-transformed data
    xtrain_log_mean = np.mean(log_transform(Xtrain), axis=0)
    xtrain_log_std = np.std(log_transform(Xtrain), axis=0)
    ytrain_log_mean = np.mean(log_transform(yTrain), axis=0)
    ytrain_log_std = np.std(log_transform(yTrain), axis=0)
    
    # Store all stats for use in transformations
    stats = {
        'xtrain_mean': xtrain_mean,
        'xtrain_std': xtrain_std,
        'ytrain_mean': ytrain_mean,
        'ytrain_std': ytrain_std,
        'log_mean': xtrain_log_mean,
        'log_std': xtrain_log_std,
        'ytrain_log_mean': ytrain_log_mean,
        'ytrain_log_std': ytrain_log_std
    }
    
    # Apply the chosen transformation
    if transformation_type == 'standardization':
        Xtrain_transformed = standard_transform(stats, Xtrain, 'xtrain_mean', 'xtrain_std')
        yTrain_transformed = standard_transform(stats, yTrain, 'ytrain_mean', 'ytrain_std')
        Xval_transformed = standard_transform(stats, Xval, 'xtrain_mean', 'xtrain_std')
        yval_transformed = standard_transform(stats, yval, 'ytrain_mean', 'ytrain_std')
        Xtest_transformed = standard_transform(stats, Xtest, 'xtrain_mean', 'xtrain_std')
        ytest_transformed = standard_transform(stats, ytest, 'ytrain_mean', 'ytrain_std')
    
    elif transformation_type == 'log':
        Xtrain_transformed = log_transform(Xtrain)
        yTrain_transformed = log_transform(yTrain)
        Xval_transformed = log_transform(Xval)
        yval_transformed = log_transform(yval)
        Xtest_transformed = log_transform(Xtest)
        ytest_transformed = log_transform(ytest)
        
    elif transformation_type == 'exp':
        Xtrain_transformed = exp_transform(Xtrain)
        yTrain_transformed = exp_transform(yTrain)
        Xval_transformed = exp_transform(Xval)
        yval_transformed = exp_transform(yval)
        Xtest_transformed = exp_transform(Xtest)
        ytest_transformed = exp_transform(ytest)
        
    elif transformation_type == 'log_standardization':
        Xtrain_transformed = combined_log_standard_transform(Xtrain, stats)
        yTrain_transformed = combined_log_standard_transform(yTrain, stats, y=True)
        Xval_transformed = combined_log_standard_transform(Xval, stats)
        yval_transformed = combined_log_standard_transform(yval, stats, y=True)
        Xtest_transformed = combined_log_standard_transform(Xtest, stats)
        ytest_transformed = combined_log_standard_transform(ytest, stats, y=True)
        
    else:
        raise ValueError(f"Unknown transformation type: {transformation_type}")
    
    return Xtrain_transformed, yTrain_transformed, Xval_transformed, yval_transformed, Xtest_transformed, ytest_transformed, stats
