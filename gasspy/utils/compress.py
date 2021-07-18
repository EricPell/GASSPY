""" Compress a parameter """
import numpy as np

def number(input_number, reduction_parameters):
    (decimal_precision, compression_ratio) = reduction_parameters
    """Reduce numeric resolution of array to at a given decimal precision by a compression factor"""
    scaled_number = input_number / compression_ratio
    truncated_scaled_number = np.around(scaled_number, decimals = decimal_precision)
    return np.around(truncated_scaled_number * compression_ratio, decimals = decimal_precision)
    #return np.around((input_array / compression_ratio), dec_precision) * compression_ratio

def array(input_array, reduction_parameters):
    (decimal_precision, compression_ratio) = reduction_parameters
    """Reduce numeric resolution of array to at a given decimal precision by a compression factor"""
    scaled_array = np.divide(input_array, compression_ratio)

    truncated_scaled_array = np.around(scaled_array, decimal_precision)
    compressed_array = np.around(np.multiply(truncated_scaled_array, compression_ratio), decimal_precision)

    return compressed_array
    #return np.around((input_array / compression_ratio), dec_precision) * compression_ratio

def dictionary(input_dict, reduction_parameters):
    (decimal_precision, compression_ratio) = reduction_parameters
    """Reduce numeric resolution of array to at a given decimal precision by a compression factor"""
    compressed_dict = {}
    for key in input_dict:
        scaled = input_dict[key] / compression_ratio[key]
        truncated = np.around(scaled, decimals = decimal_precision)
        compressed_dict[key] = np.around(truncated*compression_ratio[key])
    return compressed_dict
    #return np.around((input_array / compression_ratio), dec_precision) * compression_ratio
