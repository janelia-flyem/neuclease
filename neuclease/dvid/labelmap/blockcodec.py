"""
Numba implementation of DVID's labelarray block codec,
replacing the libdvid-cpp functions encode_label_block() and
DVIDNodeService.inflate_labelarray_blocks3D_from_raw().
"""
import zlib
import numpy as np
from numba import njit

BW = 64      # block width
SBW = 8      # subblock width
G = BW // SBW
NSB = G**3   # number of subblocks
SBV = SBW**3  # voxels per subblock


@njit(nogil=True, cache=True)
def _bit_length(n):
    # ceil(log2(n)), with 0 for n <= 1
    b = 0
    while (1 << b) < n:
        b += 1
    return b


@njit(nogil=True, cache=True)
def decode_label_block_into(enc, out):
    """
    Decode a single (already-decompressed) labelarray block into
    ``out``, which must be a (64,64,64) uint64 array (possibly a view).
    """
    if len(enc) < 16:
        raise ValueError("Encoded block is too short")
    hdr = enc[:16].view(np.uint32)
    if hdr[0] != G or hdr[1] != G or hdr[2] != G:
        raise ValueError("Unsupported subblock grid dimensions")
    N = hdr[3]
    if N == 0:
        raise ValueError("Encoded block has an empty label list")
    if len(enc) < 16 + 8*N:
        raise ValueError("Encoded block is truncated (label list)")
    pos = 16
    labels = enc[pos:pos + 8*N].view(np.uint64)
    pos += 8*N

    if N == 1:
        out[:] = labels[0]
        return

    if len(enc) < pos + 2*NSB:
        raise ValueError("Encoded block is truncated (subblock label counts)")
    Ns = enc[pos:pos + 2*NSB].view(np.uint16)
    pos += 2*NSB

    # Validate the remaining size up-front, so the loops
    # below can't read past the end of the buffer.
    total = 0
    stream_bytes = 0
    for i in range(NSB):
        if Ns[i] > SBV:
            raise ValueError("Encoded block has an invalid subblock label count")
        total += Ns[i]
        stream_bytes += 64*_bit_length(Ns[i])
    if len(enc) < pos + 4*total + stream_bytes:
        raise ValueError("Encoded block is truncated (subblock indices or voxel data)")

    indices = enc[pos:pos + 4*total].view(np.uint32)
    for i in range(total):
        if indices[i] >= N:
            raise ValueError("Encoded block has an out-of-range label index")
    pos += 4*total

    sb_labels = np.empty(SBV, np.uint64)
    idx_start = 0
    sb = 0
    for gz in range(G):
        for gy in range(G):
            for gx in range(G):
                n = Ns[sb]
                for j in range(n):
                    sb_labels[j] = labels[indices[idx_start + j]]
                idx_start += n
                sb += 1

                z0 = gz*SBW
                y0 = gy*SBW
                x0 = gx*SBW
                b = _bit_length(n)
                if b == 0:
                    v = sb_labels[0] if n == 1 else np.uint64(0)
                    for z in range(SBW):
                        for y in range(SBW):
                            for x in range(SBW):
                                out[z0+z, y0+y, x0+x] = v
                    continue

                # MSB-first bitstream.  512*b bits is always a whole number of bytes.
                mask = (1 << b) - 1
                bitpos = 0
                for z in range(SBW):
                    for y in range(SBW):
                        for x in range(SBW):
                            byte = pos + (bitpos >> 3)
                            word = np.uint32(enc[byte]) << 8
                            if byte + 1 < len(enc):
                                word |= np.uint32(enc[byte+1])
                            shift = 16 - b - (bitpos & 7)
                            out[z0+z, y0+y, x0+x] = sb_labels[(word >> shift) & mask]
                            bitpos += b
                pos += 64*b


@njit(nogil=True, cache=True)
def _parse_blocks_stream(buf):
    """
    Parse the /blocks stream headers.
    Returns (N,3) block coords (ZYX, in block units), and (N,2) [start, stop) byte ranges.
    """
    n = 0
    pos = 0
    while pos < len(buf):
        if pos + 16 > len(buf):
            raise ValueError("Blocks stream is truncated (block header)")
        nbytes = buf[pos:pos+16].view(np.int32)[3]
        if nbytes < 0 or pos + 16 + nbytes > len(buf):
            raise ValueError("Blocks stream is truncated (block data)")
        pos += 16 + nbytes
        n += 1
    coords = np.empty((n, 3), np.int64)
    ranges = np.empty((n, 2), np.int64)
    pos = 0
    for i in range(n):
        h = buf[pos:pos+16].view(np.int32)
        coords[i, 0] = h[2]
        coords[i, 1] = h[1]
        coords[i, 2] = h[0]
        ranges[i, 0] = pos + 16
        ranges[i, 1] = pos + 16 + h[3]
        pos += 16 + h[3]
    return coords, ranges


def inflate_labelarray_blocks(buf, shape, offset):
    """
    Inflate a labelmap ``GET /blocks`` payload (a stream of gzipped
    labelarray blocks, each preceded by a 16-byte header) into a volume.

    Args:
        buf:
            bytes, as returned by the ``/blocks`` endpoint.
        shape:
            Shape (ZYX) of the output volume.
        offset:
            Voxel coordinate (ZYX) of the output volume's first voxel.
            Every block in the stream must lie entirely within the volume.

    Returns:
        uint64 ndarray of the given shape.
        Voxels not covered by any block are 0.
    """
    shape = np.asarray(shape)
    offset = np.asarray(offset)
    a = np.frombuffer(buf, np.uint8)
    coords, ranges = _parse_blocks_stream(a)
    out = np.zeros(tuple(shape), np.uint64)
    for (bz, by, bx), (start, stop) in zip(coords, ranges):
        if stop == start:
            continue
        corner = np.array((bz, by, bx)) * BW - offset
        if (corner < 0).any() or (corner + BW > shape).any():
            raise ValueError(
                f"Block at {(corner + offset).tolist()} (ZYX) lies outside the "
                f"volume at offset {offset.tolist()} with shape {shape.tolist()}")
        z, y, x = corner
        enc = np.frombuffer(zlib.decompress(a[start:stop], 31), np.uint8)
        decode_label_block_into(enc, out[z:z+BW, y:y+BW, x:x+BW])
    return out


@njit(nogil=True, cache=True)
def _encode_label_block(block):
    """
    Encode a (64,64,64) uint64 block.  Returns uint8 array (not gzipped).
    """
    # Per-subblock tables (unique labels, in order of first appearance)
    # and per-voxel indices into those tables.
    sb_tables = np.empty((NSB, SBV), np.uint64)
    sb_counts = np.zeros(NSB, np.int32)
    voxel_idx = np.empty((NSB, SBV), np.uint16)

    sorted_buf = np.empty(SBV, np.uint64)
    sb = 0
    for gz in range(G):
        for gy in range(G):
            for gx in range(G):
                z0 = gz*SBW
                y0 = gy*SBW
                x0 = gx*SBW
                table = sb_tables[sb]
                n = 0
                last = np.uint64(0)
                last_i = -1
                i = 0
                for z in range(SBW):
                    for y in range(SBW):
                        for x in range(SBW):
                            v = block[z0+z, y0+y, x0+x]
                            # Fast path: same as previous voxel
                            if last_i >= 0 and v == last:
                                voxel_idx[sb, i] = last_i
                                i += 1
                                continue
                            # Linear search (subblocks usually have few labels)
                            k = 0
                            while k < n and table[k] != v:
                                k += 1
                            if k == n:
                                table[n] = v
                                n += 1
                            voxel_idx[sb, i] = k
                            last = v
                            last_i = k
                            i += 1
                sb_counts[sb] = n
                sb += 1

    # Global table: unique over all subblock tables
    total = 0
    for s in range(NSB):
        total += sb_counts[s]
    all_labels = np.empty(total, np.uint64)
    p = 0
    for s in range(NSB):
        all_labels[p:p+sb_counts[s]] = sb_tables[s, :sb_counts[s]]
        p += sb_counts[s]
    global_labels = np.unique(all_labels)
    N = len(global_labels)

    # Compute output size
    size = 16 + 8*N
    if N > 1:
        size += 2*NSB + 4*total
        for s in range(NSB):
            size += 64*_bit_length(sb_counts[s])

    out = np.zeros(size, np.uint8)
    hdr = out[:16].view(np.uint32)
    hdr[0] = G
    hdr[1] = G
    hdr[2] = G
    hdr[3] = N
    pos = 16
    out[pos:pos+8*N].view(np.uint64)[:] = global_labels
    pos += 8*N
    if N == 1:
        return out

    Ns = out[pos:pos+2*NSB].view(np.uint16)
    for s in range(NSB):
        Ns[s] = sb_counts[s]
    pos += 2*NSB

    gidx = np.searchsorted(global_labels, all_labels).astype(np.uint32)
    out[pos:pos+4*total].view(np.uint32)[:] = gidx
    pos += 4*total

    for s in range(NSB):
        b = _bit_length(sb_counts[s])
        if b == 0:
            continue
        bitpos = 0
        for i in range(SBV):
            v = np.uint32(voxel_idx[s, i])
            shift = 16 - b - (bitpos & 7)
            word = v << shift
            byte = pos + (bitpos >> 3)
            out[byte] |= np.uint8(word >> 8)
            if shift < 8:
                out[byte+1] |= np.uint8(word & 0xFF)
            bitpos += b
        pos += 64*b
    return out


def encode_label_block(block):
    """
    Encode a (64,64,64) label block using DVID's labelarray block encoding.
    Returns bytes (not gzipped).
    """
    block = np.asarray(block, np.uint64)
    if block.shape != (64, 64, 64):
        raise ValueError(f"Can't encode block: Bad shape: {block.shape}")
    return _encode_label_block(block).tobytes()
