import numpy as np


def _encode_uint64_series(s, dtype='<u8'):
    """
    Encode a pandas Series (or Index) of N values
    into a numpy array of N buffers (bytes objects).
    """
    id_buf = s.to_numpy(dtype).tobytes()
    id_bufs = [
        id_buf[offset:(offset+8)]
        for offset in range(0, len(id_buf), 8)
    ]
    return np.array(id_bufs, dtype=object)
