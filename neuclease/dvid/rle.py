from itertools import product
import numpy as np
from numba import jit
from numba.types import int32
from numba.typed import List

import pandas as pd


from neuclease.util import lexsort_columns

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


def construct_rle_payload(coords_zyx):
    """
    Construct the RLE payload to send to DVID's split endpoint,
    including the header bytes.
    """
    coords_zyx = coords_zyx.astype(np.int32, copy=False)
    rles_zyx = runlength_encode_to_lengths(coords_zyx, assume_sorted=False)
    rles_xyz = rles_zyx[:, (2,1,0,3)]

    payload_items = []
    payload_items.append( np.array([0, 3, 0, 0], dtype=np.uint8) )
    payload_items.append( np.array([0, len(rles_xyz)], dtype=np.uint32) )
    payload_items.append( rles_xyz )

    payload = b''.join( list(map(bytes, payload_items)) )
    return payload


def combine_sparsevol_rle_responses(rle_payloads):
    """
    Combined two sets of RLE payloads (perhaps fetched from DVID),
    into a single RLE payload.
    """
    combined_coords = map(parse_rle_response, rle_payloads)
    combined_coords = np.concatenate([*combined_coords])
    return construct_rle_payload(combined_coords)


def parse_rle_response(response_bytes, dtype=np.int32, format='coords'):  # @ReservedAssignment
    """
    Parse a (legacy) RLE response from DVID, used by various endpoints
    such as 'sparsevol' and 'sparsevol-coarse'.

    Args:
        response_bytes:
            RLE bytes as returned by a DVID endpoint, e.g. /sparsevol

        dtype:
            The dtype of the returned coordinate array.
            Must be either np.int32 (the default) or np.int16.
            If you know the results will not exceed 2**15 in any coordinate,
            you can save some RAM by selecting np.int16

        format:
            Either 'rle', 'ranges', or 'coords'.  See return value explanation.

    Return:
        If format == 'rle', returns the RLE start coordinates and RLE lengths as two arrays:

            (start_coords, lengths)

            where start_coords is in the form:

                [[Z,Y,X], [Z,Y,X], ...]

            and lengths is a 1-D array:

                [length, length, ...]

        If format == 'ranges':
            Return the RLEs as ranges, in the form:

                [[Z,Y,X0,X1], [Z,Y,X0,X1], ...]

            Note: By DVID conventions, the interval [X0,X1] is inclusive,
                  i.e. X1 is IN the range -- not one beyond the range,
                  which would normally be the Python convention.

        If format == 'coords', returns an array of coordinates of the form:

            [[Z,Y,X],
             [Z,Y,X],
             [Z,Y,X],
             ...
            ]
    """
    assert isinstance(response_bytes, bytes)
    assert dtype in (np.int32, np.int16)
    assert format in ('coords', 'rle', 'ranges')
    descriptor = response_bytes[0]
    ndim = response_bytes[1]
    run_dimension = response_bytes[2]

    assert descriptor == 0, f"Don't know how to handle this payload. (descriptor: {descriptor})"
    assert ndim == 3, "Expected XYZ run-lengths"
    assert run_dimension == 0, "This function assumes the RLE run dimension is X"

    content_as_int32 = np.frombuffer(response_bytes, np.int32)
    _voxel_count = content_as_int32[1]  # noqa
    run_count = content_as_int32[2]
    rle_items = content_as_int32[3:].reshape(-1,4)

    assert len(rle_items) == run_count, \
        f"run_count ({run_count}) doesn't match data array length ({len(rle_items)})"

    rle_starts_xyz = rle_items[:,:3]
    rle_starts_zyx = rle_starts_xyz[:,::-1]
    rle_lengths = rle_items[:,3]

    # For now, DVID always returns a voxel_count of 0, so we can't make this assertion.
    # assert rle_lengths.sum() == _voxel_count,\
    #    f"Voxel count ({voxel_count}) doesn't match expected sum of run-lengths ({rle_lengths.sum()})"

    if dtype == np.int16:
        assert rle_starts_zyx[:, :2].min() >= -(2**15), "Can't return np.int16 -- result would overflow"
        assert rle_starts_zyx[:, :2].max() < 2**15, "Can't return np.int16 -- result would overflow"
        assert (rle_starts_zyx[:, 2] + rle_lengths).min() >= -(2**15), "Can't return np.int16 -- result would overflow"
        assert (rle_starts_zyx[:, 2] + rle_lengths).max() < 2**15, "Can't return np.int16 -- result would overflow"
        rle_starts_zyx = rle_starts_zyx.astype(np.int16)
        rle_lengths = rle_lengths.astype(np.int16)

    if format == 'ranges':
        ranges = np.zeros((len(rle_items), 4), dtype)
        ranges[:, :3] = rle_starts_zyx
        ranges[:, 3] = rle_starts_zyx[:,2] + rle_lengths - 1  # end is INCLUSIVE, as noted above
        return ranges

    if format == 'rle':
        return rle_starts_zyx, rle_lengths

    if format == 'coords':
        # Sadly, the decode function requires contiguous arrays, so we must copy.
        rle_starts_zyx = rle_starts_zyx.copy('C')
        rle_lengths = rle_lengths.copy('C')

        dense_coords = runlength_decode_from_lengths(rle_starts_zyx, rle_lengths)
        assert dense_coords.dtype == dtype

        assert rle_lengths.sum() == len(dense_coords), "Got the wrong number of coordinates!"
        return dense_coords


def rle_box_dilation(start_coords, lengths, radius):
    """
    Dilate the given RLEs by some radius, using simple
    "box" dilation (not a proper spherical dilation).

    Equivalent to decoding the given RLEs, constructing a binary mask,
    dilating it with a rectangular structuring element,
    and then re-encoding the resulting nonzero coordinates.

    But much faster than that, and requiring much less RAM.

    Args:
        start_coords, lengths:
            See runlength_decode_from_lengths()

        radius:
            The "radius" of the cube-shaped structuring element used to dilate.
    """
    assert start_coords.ndim == 2
    assert lengths.ndim == 1
    assert len(start_coords) == len(lengths)

    if len(start_coords) == 0:
        return np.zeros((0, 3), np.int32), np.zeros((0,), np.int32)

    # Concatenate: (Z,Y,X,L)
    table = np.concatenate((start_coords, lengths[:, None]), axis=1)

    # Dilation in X is trivial
    table[:] += (0, 0, -radius, 2*radius)

    # Dilate in Z
    new_tables = []
    for rz in range(-radius, radius+1):
        new_tables.append( table + np.int32((rz,0,0,0)) )

    # Choose chunk size for condense step.
    # As a heuristic, anything smaller than the size of the
    # original RLE seems like a reasonable chunk size to use.
    # Smaller chunks require less RAM usage overall,
    # but if there are too many chunks the code isn't well vectorized.
    chunk_size = max(1024, len(start_coords) // 8)

    # Condense.
    table = np.concatenate(new_tables)
    table = _condense_rles(table, chunk_size)

    # Dilate in Y
    new_tables = []
    for ry in range(-radius, radius+1):
        new_tables.append( table + np.int32((0,ry,0,0)) )

    # Condense.
    table = np.concatenate(new_tables)
    table = _condense_rles(table, chunk_size)

    return table[:, :3], table[:, 3]


def _condense_rles(table, chunk_size):
    """
    Given an RLE table (columns Z,Y,X,L) which may contain overlapping RLEs,
    condense it into a minmal RLE table.

    This is done by decoding it, dropping duplicate coordinates, and re-encoding it.
    The table is processed in chunks to avoid too much instantaneous RAM usage.
    """
    assert len(table) > 0

    # Sort by Z,Y columns
    order = np.lexsort(table[:, :2].transpose()[::-1])
    table = table[order]

    # We could do this as one big decode step and drop duplicates afterward,
    # but that would require a lot of RAM.
    # Instead, break into chunks and drop duplicates as we go.
    condensed_chunks = []
    start = 0
    while start < len(table):
        # Search for a stop row that splits *between* ZY groups
        # TODO: This would be faster without an explicit loop, using diff or something
        stop = start+chunk_size
        while stop < len(table) and (table[stop, :2] == table[stop-1, :2]).all():
            stop += 1

        # Must copy because C-order is required
        chunk = table[start:stop]
        start = stop

        chunk_coords = runlength_decode_from_lengths( chunk[:, :3].copy(), chunk[:, 3].copy() )
        chunk_coords = pd.DataFrame(chunk_coords, columns=[*'zyx'])
        chunk_coords = chunk_coords.drop_duplicates().sort_values([*'zyx'])
        chunk_coords = np.asarray(chunk_coords.values, order='C')

        condensed_chunk = runlength_encode_to_lengths(chunk_coords, True)
        condensed_chunks.append( condensed_chunk )
        del chunk_coords

    condensed_table = np.concatenate( condensed_chunks )
    return condensed_table


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

    runs = _runlength_encode_to_ranges(coord_list_zyx)
    return np.asarray(runs, coord_list_zyx.dtype).reshape((-1,4))


@jit(nopython=True, nogil=True)
def _runlength_encode_to_ranges(coord_list_zyx):
    """
    Helper function for runlength_encode(), above.

    Args:
        coord_list_zyx:
            Array of shape (N,3), of form [[Z,Y,X], [Z,Y,X], ...],
            pre-sorted in Z-Y-X order.  Duplicates permitted.

    Returns:
        Flattened runlength encoding, as a numba typed.List:

            [Z,Y,X1,X2,Z,Y,X1,X2,Z,Y,X1,X2,...]

        (Must be converted to np.array and un-flattened by the caller.)
    """
    runs = List.empty_list(item_type=int32)

    # Start the first run
    (prev_z, prev_y, prev_x) = current_run_start = coord_list_zyx[0]

    for i in range(1, len(coord_list_zyx)):
        (z,y,x) = coord = coord_list_zyx[i]

        # If necessary, end the current run and start a new one
        # (Also, support duplicate coords without breaking the current run.)
        if (z != prev_z) or (y != prev_y) or (x != prev_x and x != 1+prev_x):
            runs.extend(current_run_start)
            runs.append(prev_x)
            current_run_start = coord

        (prev_z, prev_y, prev_x) = (z,y,x)

    # End the last run
    runs.extend(current_run_start)
    runs.append(prev_x)
    return runs


def runlength_encode_mask_to_ranges(mask, mask_box=None):
    """
    Given a binary mask, return an array of
    run-length encodings of the form:

        [[Z,Y,X1,X2],
         [Z,Y,X1,X2],
         [Z,Y,X1,X2],
         ...
        ]

    Note:
        The interval [X1,X2] is INCLUSIVE, following DVID conventions,
        not Python conventions (i.e. X2 is not one-past-the-end).

    Args:
        mask:
            binary volume, 3D

        mask_box:
            If provided, should correspond to the location of the mask in global coordinates.
            The returned ranges will be offset to reflect global coordinates.

    Returns:
        Ranges array as described above, shape (N,4)
    """
    assert mask.ndim == 3
    runs = _runlength_encode_mask_to_ranges(mask)
    ranges = np.asarray(runs).reshape((-1,4))

    if mask_box is not None:
        mask_box = np.asarray(mask_box)
        assert mask_box.shape == (2,3)
        assert mask.shape == tuple(mask_box[1] - mask_box[0]), \
            "mask shape doesn't correspond to mask_box"

        ranges[:] += mask_box[0, (0,1,2,2)]
        assert (ranges >= mask_box[0, (0,1,2,2)]).all()
        assert (ranges <= mask_box[1, (0,1,2,2)]).all()

    return ranges


@jit(nopython=True)
def _runlength_encode_mask_to_ranges(mask):
    """
    Helper function for runlength_encode_mask_to_ranges(), above.

    Args:
        mask:
            binary volume, 3D

    Returns:
        Flattened runlength encoding, as a numba typed.List:

            [Z,Y,X1,X2,Z,Y,X1,X2,Z,Y,X1,X2,...]

        Must be converted to np.array and un-flattened by the caller.
        Uses DVID conventions for the ranges, i.e. X1,X2 is inclusive,
        not one-past-the-end.
    """
    runs = List.empty_list(item_type=int32)

    Z, Y, X = mask.shape
    for z in np.arange(Z, dtype=np.int32):
        for y in np.arange(Y, dtype=np.int32):
            x0 = x1 = int32(0)
            in_run = False
            for x in np.arange(X, dtype=np.int32):
                if not in_run and mask[z,y,x]:
                    x0 = x
                    in_run = True
                elif in_run and not mask[z,y,x]:
                    x1 = np.int32(x-1)
                    in_run = False
                    runs.extend([z, y, x0, x1])

            if in_run:
                x1 = np.int32(X-1)
                runs.extend([z, y, x0, x1])

    return runs


def runlength_encode_to_lengths(coord_list_zyx, assume_sorted=False):
    """
    Given an array of coordinates in the form:

        [[Z,Y,X],
         [Z,Y,X],
         [Z,Y,X],
         ...
        ]

    Return an array of run-length encodings of the form:

        [[Z,Y,X,N],
         [Z,Y,X,N],
         [Z,Y,X,N],
         ...
        ]

    ... where N is the length of each run.

    Args:
        coord_list_zyx:
            Array of shape (C,3)

        assume_sorted:
            If True, the provided coordinates are assumed to be pre-sorted in Z-Y-X order.
            Otherwise, this function sorts them before the RLEs are computed.    """
    if len(coord_list_zyx) == 0:
        return np.ndarray( (0,4), np.int32 )

    coord_list_zyx = np.asarray(coord_list_zyx)
    assert coord_list_zyx.ndim == 2
    assert coord_list_zyx.shape[1] == 3

    if not assume_sorted:
        sorting_ind = np.lexsort(coord_list_zyx.transpose()[::-1])
        coord_list_zyx = coord_list_zyx[sorting_ind]

    # Compute ranges and then convert to lengths.
    runs = _runlength_encode_to_ranges(coord_list_zyx)
    rle_result = np.array(runs, coord_list_zyx.dtype).reshape((-1,4))
    rle_result[:,3] = 1 + rle_result[:,3] - rle_result[:,2]
    return rle_result


@jit(["i8[:,:](i8[:,::1],i8[::1])", "i4[:,:](i4[:,::1],i4[::1])", "i2[:,:](i2[:,::1],i2[::1])"], nopython=True, nogil=True)  # See note about signature, below.
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

    c = 0  # coord row
    for i in range(len(rle_start_coords_zyx)):
        (z, y, x0) = rle_start_coords_zyx[i]
        length = rle_lengths[i]
        for x in range(x0, x0+length):
            coords[c, 0] = z
            coords[c, 1] = y
            coords[c, 2] = x
            c += 1

    return coords


def runlength_decode_from_ranges_to_mask(rle_array_zyx, mask_box=None):
    """
    Used for parsing the result of DVID's /roi endpoint.

    Args:
        Array of run-length encodings in the form:

        [[Z,Y,X1,X2],
         [Z,Y,X1,X2],
         [Z,Y,X1,X2],
         ...
        ]

    Note: The interval [X1,X2] is INCLUSIVE, following DVID conventions, not Python conventions.

    Returns:
        (mask, mask_box)

        Where mask is a binary mask of the ROI, in which each voxel
        represents one ROI block (scale 5).
        The mask will be cropped to the bounding box of the ROI,
        and mask_box is the bounding-box of the mask, in scale-5 units.
    """
    rle_array_zyx = np.asarray(rle_array_zyx, dtype=np.int32)
    if mask_box is None:
        mask_box = rle_ranges_box(rle_array_zyx)

    mask_box = np.asarray(mask_box, dtype=np.int32)
    mask_shape = tuple((mask_box[1] - mask_box[0]).tolist())

    # Switch from inclusive conventions for python conventions (one-past-the-end)
    ranges_array = rle_array_zyx + np.asarray([0,0,0,1], dtype=np.int32)

    # Offset to mask-local coordinates
    ranges_array -= mask_box[0, (0,1,2,2)]

    # Crop the X ranges to the mask_shape edges.
    # Out-of-bounds ranges will end up with X1 >= X2 (and discarded below).
    ranges_array[:, 2] = np.maximum(ranges_array[:, 2], 0)
    ranges_array[:, 3] = np.minimum(ranges_array[:, 3], mask_shape[2])

    # Discard ranges that are out of bounds in Z/Y, or were cropped to nothing in X
    keep =  (ranges_array[:, :2] >= 0).all(axis=1)
    keep &= (ranges_array[:, :2] < mask_shape[:2]).all(axis=1)
    keep &= (ranges_array[:, 2] < ranges_array[:, 3])

    ranges_array = np.asarray(ranges_array[keep, :], order='C')
    mask = np.zeros(mask_shape, dtype=np.uint8)
    _write_mask_from_ranges(ranges_array, mask)
    return mask.astype(bool), mask_box


def rle_ranges_box(rle_array_zyx):
    """
    Given a list of run-length encodings from DVID's /roi endpoint,
    return the bounding-box of the encoded sparse volume.

    Args:
        Array of run-length encodings in the form:

        [[Z,Y,X1,X2],
         [Z,Y,X1,X2],
         [Z,Y,X1,X2],
         ...
        ]

    Note:
        The input interval [X1,X2] is INCLUSIVE, following DVID conventions,
        not Python conventions. But this function returns a bounding box
        according to the usual conventions, i.e. [start, stop], where stop
        is one-past-the-end.

    Returns:
        bounding_box (start, stop)
    """
    rle_array_zyx = np.asarray(rle_array_zyx, dtype=np.int32)
    min_coord =   rle_array_zyx[:, (0,1,2)].min(axis=0)
    max_coord = 1+rle_array_zyx[:, (0,1,3)].max(axis=0)
    mask_box = np.asarray([min_coord, max_coord])
    return mask_box


@jit(nopython=True)
def _write_mask_from_ranges(ranges_array, mask):
    """
    Fill the given mask using the given RLE ranges.
    Note: Here, we assume EXCLUSIVE conventions,
    i.e. [Z, Y, X0, X1], where X1 is one-past-the-end of the range.
    """
    for i in range(len(ranges_array)):
        (z, y, x0, x1) = ranges_array[i]
        mask[z, y, x0:x1] = 1


@jit(["i8[:,:](i8[:,::1])", "i4[:,:](i4[:,::1])", "i2[:,:](i2[:,::1])"], nopython=True, nogil=True)  # See note about signature, below.
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

    c = 0  # coord row
    for i in range(len(rle_array_zyx)):
        (z, y, x1, x2) = rle_array_zyx[i]
        for x in range(x1, x2+1):
            coords[c, 0] = z
            coords[c, 1] = y
            coords[c, 2] = x
            c += 1

    return coords


@jit(nopython=True, nogil=True)
def _split_x_ranges_for_grid(ranges, block_shape, halo=0):
    """
    Given RLEs encodings in the form of 'ranges' as returned by DVID:

        [[Z,Y,X1,X2],
         [Z,Y,X1,X2],
         [Z,Y,X1,X2],
         ...
        ]

    Note: The range INCLUDES X2, unlike numpy conventions.

    Split the RLEs at block bounaries...
    """
    BZ, BY, BX = block_shape

    BX1 = ranges[:, 2] // BX
    BX2 = ranges[:, 3] // BX
    num_split_ranges = (BX2 - BX1 + 1).sum()

    # [Bz, By, Bx, z, y, x1, x2]
    results = np.empty((num_split_ranges, 7), dtype=np.int32)
    j = 0
    for i in range(len(ranges)):
        z, y, x1, x2 = ranges[i]
        Bz, By, Bx = ranges[i, :3] // block_shape

        xa, xb = x1, x2
        while (xa // BX) != (x2 // BX):
            xb = (xa // BX + 1) * BX - 1
            results[j, :] = [Bz, By, Bx, z, y, max(x1, xa - halo), min(x2, xb + halo)]
            j += 1
            xa = xb + 1
            Bx = (xa // BX)

        results[j, :] = [Bz, By, Bx, z, y, max(x1, xa - halo), x2]
        j += 1

    assert j == num_split_ranges
    return results


def split_ranges_for_grid(ranges, block_shape, halo=0):
    block_shape = np.array(block_shape)
    grid_ranges = _split_x_ranges_for_grid(ranges, block_shape, halo)
    df = pd.DataFrame(grid_ranges, columns=['Bz', 'By', 'Bx', 'z', 'y', 'x1', 'x2'])
    if halo == 0:
        return df

    BZ, BY, BX = block_shape

    # For speed, exclude the purely non-halo ranges since they aren't needed.
    halo_ranges_df = df.query(
        'not (Bz * @BZ + @halo <= z and z < (Bz+1) * @BZ - @halo and '
        '     By * @BY + @halo <= y and y < (By+1) * @BY - @halo)')

    halo_dfs = {}
    halo_dfs['Z-upper'] = halo_ranges_df.query('z < Bz * @BZ + @halo').copy()
    halo_dfs['Z-upper'][['Bz', 'By']] -= (1, 0)

    halo_dfs['Y-upper'] = halo_ranges_df.query('y < By * @BY + @halo').copy()
    halo_dfs['Y-upper'][['Bz', 'By']] -= (0, 1)

    halo_dfs['Z-lower'] = halo_ranges_df.query('z >= (Bz+1) * @BZ - @halo').copy()
    halo_dfs['Z-lower'][['Bz', 'By']] += (1, 0)

    halo_dfs['Y-lower'] = halo_ranges_df.query('y >= (By+1) * @BY - @halo').copy()
    halo_dfs['Y-lower'][['Bz', 'By']] += (0, 1)

    # halo_dfs['Z-upper-Y-upper'] = df.query('z + 1 - Bz * @BZ < @halo').copy()
    # halo_dfs['Z-upper-Y-upper'][['Bz', 'By']] -= (1, 0)

    # Combine, then use an inner merge to drop any blocks ended up with only halos (no interior portion).
    halo_df = pd.concat(halo_dfs.values(), ignore_index=True)
    halo_df = df[['Bz', 'By', 'Bx']].drop_duplicates().merge(halo_df, 'inner', on=['Bz', 'By', 'Bx'])

    df = pd.concat((df, halo_df), ignore_index=True)
    df.sort_values(['Bz', 'By', 'Bx', 'z', 'y'], ignore_index=True, inplace=True)
    return df


    # for dz, dy in product((-1, 0, 1), (-1, 0, 1)):
    #     if (dz, dy) == (0,0):
    #         continue
    #     halo_df[['Bz', 'By']] = df[['Bz', 'By']] - (dz, dy)
    #     halo_df.query('(z - Bz*BZ) - @halo')
    #     halo_df = df[['Bz', 'By', 'Bx']].merge(halo_df, 'inner', on=['Bz', 'By', 'Bx'])



def blockwise_masks_from_ranges(ranges, block_shape, halo=0):
    block_shape = np.asarray(block_shape)
    full_block_shape = block_shape + 2*halo
    assert len(block_shape) == 3
    BZ, BY, BX = block_shape

    ranges_df = split_ranges_for_grid(ranges, block_shape, halo)
    coords = ranges_df[['Bz', 'By', 'Bx']].drop_duplicates().values
    masks = np.zeros((len(coords), *full_block_shape), dtype=bool)

    boxes = np.array([block_shape * coords - halo,
                      block_shape * (coords + 1) + halo])
    boxes = boxes.transpose(1, 0, 2)

    for i, ((Bz, By, Bx), block_df) in enumerate(ranges_df.groupby(['Bz', 'By', 'Bx'], sort=False)):
        block_ranges = block_df[['z', 'y', 'x1', 'x2']].values

        Z_offset = Bz * BZ - halo
        Y_offset = By * BY - halo
        X_offset = Bx * BX - halo

        # Offset to local coordinates,
        # and switch X2 to EXCLUSIVE convention as expected by _write_mask_from_ranges()
        block_ranges -= (Z_offset, Y_offset, X_offset, X_offset - 1)
        _write_mask_from_ranges(block_ranges, masks[i])

    assert i == len(masks) - 1
    if halo == 0:
        assert masks.sum() == (1 + ranges_df['x2'] - ranges_df['x1']).sum()
    return boxes, masks
