""" Compress a parameter """
import numpy as np

def number(input_number, decimal_precision, compression_ratio):
    """Reduce numeric resolution of array to at a given decimal precision by a compression factor"""
    return np.around((input_number / compression_ratio), decimal_precision) * compression_ratio
    #return np.around((input_array / compression_ratio), dec_precision) * compression_ratio

def array(input_array, decimal_precision, compression_ratio):
    """Reduce numeric resolution of array to at a given decimal precision by a compression factor"""
    scaled_array = np.divide(input_array, compression_ratio)
    truncated_scaled_array = np.around(scaled_array, decimal_precision)
    compressed_array = np.around(np.multiply(truncated_scaled_array, compression_ratio), decimal_precision)

    return compressed_array
    #return np.around((input_array / compression_ratio), dec_precision) * compression_ratio
