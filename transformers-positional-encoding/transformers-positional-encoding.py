import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    result = []
    for i in range(seq_length):
        sub_result = []
        for j in range(d_model):
            if j%2 == 0:
                sub_result.append(
                    np.sin(
                        i / np.power(10000, (2*i)/d_model)
                    )
                )
            else:
                sub_result.append(
                    np.cos(
                        i / np.power(10000, (2*i)/d_model)
                    )
                )
        result.append(sub_result)
    
    return np.array(result)