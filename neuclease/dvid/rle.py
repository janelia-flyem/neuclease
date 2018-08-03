import numpy as np
from numba import jit

def extract_rle_size_and_first_coord(rle_payload_bytes):
    """
    Given a binary RLE payload as returned by the /sparsevol endpoint,
    extract the count of voxels in the RLE and the first coordinate in the RLE. 
    
    Args:
        rle_payload_bytes:
            Bytes. Must be in DVID's "Legacy RLEs" format.

    Useful for sampling label value under a given RLE geometry
    (assuming all of the points in the RLE share the same label).
    
    Returns:
        voxel_count, coord_zyx
    """
    assert (len(rle_payload_bytes) - 3*4) % (4*4) == 0, \
        "Payload does not appear to be an RLE payload as defined by DVID's 'Legacy RLE' format."
    rles = np.frombuffer(rle_payload_bytes, dtype=np.uint32)[3:]
    rles = rles.reshape(-1, 4)
    first_coord_xyz = rles[0, :3]
    first_coord_zyx = first_coord_xyz[::-1]

    voxel_count = rles[:, 3].sum()
    return voxel_count, first_coord_zyx


def parse_rle_response(response_bytes, dtype=np.int32):
    """
    Parse a (legacy) RLE response from DVID, used by various endpoints
    such as 'sparsevol' and 'sparsevol-coarse'.
    
    Args:
        response_bytes:
            RLE bytes as returned by a DVID endpoint, e.g. /sparsevol
        
        dtype:
            The dtype of the returned coordinate array.
            Must be either np.int32 (the default) or np.int16.
            If you know the results will not exceed 2**16 in any coordinate,
            you can save some RAM by selecting np.int16

    Return:
        An array of coordinates of the form:

        [[Z,Y,X],
         [Z,Y,X],
         [Z,Y,X],
         ...
        ]
    """
    assert dtype in (np.int32, np.int16)
    descriptor = response_bytes[0]
    ndim = response_bytes[1]
    run_dimension = response_bytes[2]

    assert descriptor == 0, f"Don't know how to handle this payload. (descriptor: {descriptor})"
    assert ndim == 3, "Expected XYZ run-lengths"
    assert run_dimension == 0, "This function assumes the RLE run dimension is X"

    content_as_int32 = np.frombuffer(response_bytes, np.int32)
    _voxel_count = content_as_int32[1]
    run_count = content_as_int32[2]
    rle_items = content_as_int32[3:].reshape(-1,4)

    assert len(rle_items) == run_count, \
        f"run_count ({run_count}) doesn't match data array length ({len(rle_items)})"

    rle_starts_xyz = rle_items[:,:3]
    rle_starts_zyx = rle_starts_xyz[:,::-1]
    rle_lengths = rle_items[:,3]

    # Sadly, the decode function requires contiguous arrays, so we must copy.
    rle_starts_zyx = rle_starts_zyx.copy('C')
    rle_lengths = rle_lengths.copy('C')

    # For now, DVID always returns a voxel_count of 0, so we can't make this assertion.
    #assert rle_lengths.sum() == _voxel_count,\
    #    f"Voxel count ({voxel_count}) doesn't match expected sum of run-lengths ({rle_lengths.sum()})"

    if dtype == np.int16:
        assert rle_starts_zyx[:2].max() < 2**16, "Can't return np.int16 -- result would overflow"
        assert (rle_starts_zyx[:,2] + rle_lengths).max() < 2**16, "Can't return np.int16 -- result would overflow"
        rle_starts_zyx = rle_starts_zyx.astype(np.int16)
        rle_lengths = rle_lengths.astype(np.int16)

    dense_coords = runlength_decode_from_lengths(rle_starts_zyx, rle_lengths)
    assert dense_coords.dtype == dtype
    
    assert rle_lengths.sum() == len(dense_coords), "Got the wrong number of coordinates!"
    return dense_coords


def runlength_encode_to_ranges(coord_list_zyx, assume_sorted=False):
    """
    Given an array of coordinates in the form:
        
        [[Z,Y,X],
         [Z,Y,X],
         [Z,Y,X],
         ...
        ]
        
    Return an array of run-length encodings of the form:
    
        [[Z,Y,X1,X2],
         [Z,Y,X1,X2],
         [Z,Y,X1,X2],
         ...
        ]
    
    Note: The interval [X1,X2] is INCLUSIVE, following DVID conventions, not Python conventions.
    
    Args:
        coord_list_zyx:
            Array of shape (N,3)
        
        assume_sorted:
            If True, the provided coordinates are assumed to be pre-sorted in Z-Y-X order.
            Otherwise, this function sorts them before the RLEs are computed.
    
    Timing notes:
        The FIB-25 'seven_column_roi' consists of 927971 block indices.
        On that ROI, this function takes 1.65 seconds, but with numba installed,
        it takes 35 ms (after ~400 ms warmup).
        So, JIT speedup is ~45x.
    """
    if len(coord_list_zyx) == 0:
        return np.ndarray( (0,4), np.int64 )

    coord_list_zyx = np.asarray(coord_list_zyx)
    assert coord_list_zyx.ndim == 2
    assert coord_list_zyx.shape[1] == 3
    
    if not assume_sorted:
        sorting_ind = np.lexsort(coord_list_zyx.transpose()[::-1])
        coord_list_zyx = coord_list_zyx[sorting_ind]

    return _runlength_encode_to_ranges(coord_list_zyx)


@jit(nopython=True)
def _runlength_encode_to_ranges(coord_list_zyx):
    """
    Helper function for runlength_encode(), above.
    
    coord_list_zyx:
        Array of shape (N,3), of form [[Z,Y,X], [Z,Y,X], ...],
        pre-sorted in Z-Y-X order.  Duplicates permitted.
    """
    # Numba doesn't allow us to use empty lists at all,
    # so we have to initialize this list with a dummy row,
    # which we'll omit in the return value
    runs = [0,0,0,0]
    
    # Start the first run
    (prev_z, prev_y, prev_x) = current_run_start = coord_list_zyx[0]
    
    for i in range(1, len(coord_list_zyx)):
        (z,y,x) = coord = coord_list_zyx[i]

        # If necessary, end the current run and start a new one
        # (Also, support duplicate coords without breaking the current run.)
        if (z != prev_z) or (y != prev_y) or (x not in (prev_x, 1+prev_x)):
            runs += list(current_run_start) + [prev_x]
            current_run_start = coord

        (prev_z, prev_y, prev_x) = (z,y,x)

    # End the last run
    runs += list(current_run_start) + [prev_x]

    # Return as 2D array
    runs = np.array(runs).reshape((-1,4))
    return runs[1:, :] # omit dummy row (see above)


@jit(["i4[:,:](i4[:,::1],i4[::1])", "i2[:,:](i2[:,::1],i2[::1])"], nopython=True) # See note about signature, below.
def runlength_decode_from_lengths(rle_start_coords_zyx, rle_lengths):
    """
    Given a 2D array of coordinates and a 1D array of runlengths, i.e.:
        
        [[Z,Y,X], [Z,Y,X], [Z,Y,X],...]

        and 
        
        [Length, Length, Length,...]

    Return an array of coordinates of the form:

        [[Z,Y,X],
         [Z,Y,X],
         [Z,Y,X],
         ...
        ]
    
    In which every run-length has been expanded into a run
    of consecutive coordinates in the result.
    That is, result.shape == (rle_lengths.sum(), 3)
    
    Note: The "runs" are expanded along the X AXIS.
    
    Note about Signature:
    
        Due to an apparent numba bug, it is dangerous to pass non-contiguous arrays to this function.
        (It returns incorrect results.)
        
        Therefore, the signature is explicitly written above to require contiguous arrays (e.g. i4[::1]),
        If you attempt to pass a non-contiguous array, you'll see an error like this:
        
            TypeError: No matching definition for argument type(s) readonly array(int32, 2d, A), readonly array(int32, 1d, C)
    """
    coords = np.empty((rle_lengths.sum(), 3), rle_start_coords_zyx.dtype)

    c = 0 # coord row
    for i in range(len(rle_start_coords_zyx)):
        (z, y, x0) = rle_start_coords_zyx[i]
        length = rle_lengths[i]
        for x in range(x0, x0+length):
            coords[c, 0] = z
            coords[c, 1] = y
            coords[c, 2] = x
            c += 1

    return coords


@jit(["i4[:,:](i4[:,::1])", "i2[:,:](i2[:,::1])"], nopython=True) # See note about signature, below.
def runlength_decode_from_ranges(rle_array_zyx):
    """
    Used for parsing the result of DVID's /roi endpoint.
    
    Given an array of run-length encodings in the form:
        
        [[Z,Y,X1,X2],
         [Z,Y,X1,X2],
         [Z,Y,X1,X2],
         ...
        ]
        
    Return an array of coordinates of the form:

        [[Z,Y,X],
         [Z,Y,X],
         [Z,Y,X],
         ...
        ]

    Note: The interval [X1,X2] is INCLUSIVE, following DVID conventions, not Python conventions.

    Note about Signature:
    
        Due to an apparent numba bug, it is dangerous to pass non-contiguous arrays to this function.
        (It returns incorrect results.)
        
        Therefore, the signature is explicitly written above to require contiguous arrays (e.g. i4[:,::1]),
        If you attempt to pass a non-contiguous array, you'll see an error like this:
        
            TypeError: No matching definition for argument type(s) readonly array(int32, 2d, A)
    """
    total_coords = 0
    for i in range(len(rle_array_zyx)):
        total_coords += 1 + rle_array_zyx[i, 3] - rle_array_zyx[i, 2]

    coords = np.empty((total_coords, 3), rle_array_zyx.dtype)

    c = 0 # coord row
    for i in range(len(rle_array_zyx)):
        (z, y, x1, x2) = rle_array_zyx[i]
        for x in range(x1, x2+1):
            coords[c, 0] = z
            coords[c, 1] = y
            coords[c, 2] = x
            c += 1

    return coords
