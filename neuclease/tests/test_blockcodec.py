import gzip
import zlib
import base64

import pytest
import numpy as np

from neuclease.dvid.labelmap.blockcodec import encode_label_block, inflate_labelarray_blocks

# Subblock label counts which exercise every bit length (0 through 9)
SUBBLOCK_LABEL_COUNTS = [1, 2, 3, 5, 17, 33, 65, 129, 257, 512]


def _fixture_block():
    """
    The block that was encoded (by libdvid-cpp) to produce LIBDVID_ENCODED_FIXTURE, below.
    The first 10 subblocks contain 1, 2, 3, ..., 512 labels; the rest are uniform.
    """
    block = np.zeros((64,64,64), np.uint64)
    v = np.arange(512, dtype=np.uint64).reshape(8,8,8)
    for i, (gz, gy, gx) in enumerate(np.ndindex(8,8,8)):
        n = SUBBLOCK_LABEL_COUNTS[i % 10] if i < 10 else 1
        block[gz*8:gz*8+8, gy*8:gy*8+8, gx*8:gx*8+8] = (v % n) + 7*(i % 3)
    return block


def _blocks_stream(corners_zyx, encoded_blocks):
    """
    Assemble encoded blocks into a stream, as returned by DVID's GET /blocks endpoint.
    """
    stream = b''
    for corner, enc in zip(corners_zyx, encoded_blocks):
        gz = gzip.compress(enc)
        z, y, x = np.asarray(corner) // 64
        stream += np.array([x, y, z, len(gz)], np.int32).tobytes() + gz
    return stream


def _random_block(labels_per_subblock, seed=0):
    rng = np.random.default_rng(seed)
    block = np.empty((64,64,64), np.uint64)
    for gz, gy, gx in np.ndindex(8,8,8):
        n = labels_per_subblock
        # Use the full uint64 range, to exercise the high bits.
        labels = rng.permutation(np.unique(rng.integers(0, 2**64 - 1, 2*n, dtype=np.uint64, endpoint=True)))[:n]
        sb = np.concatenate([labels, rng.choice(labels, 512 - n)])
        rng.shuffle(sb)
        block[gz*8:gz*8+8, gy*8:gy*8+8, gx*8:gx*8+8] = sb.reshape(8,8,8)
    return block


def test_decode_libdvid_fixture():
    """
    Decode a block that was encoded by libdvid-cpp,
    to verify compatibility with an independent implementation.
    """
    enc = zlib.decompress(base64.b64decode(LIBDVID_ENCODED_FIXTURE))
    vol = inflate_labelarray_blocks(_blocks_stream([(64, 128, 192)], [enc]), (64,64,64), (64, 128, 192))
    assert (vol == _fixture_block()).all()

    # Our encoding uses a different label table order,
    # but should be the same size as libdvid's.
    assert len(encode_label_block(_fixture_block())) == len(enc)


@pytest.mark.parametrize('labels_per_subblock', SUBBLOCK_LABEL_COUNTS)
def test_roundtrip(labels_per_subblock):
    block = _random_block(labels_per_subblock)
    enc = encode_label_block(block)
    vol = inflate_labelarray_blocks(_blocks_stream([(0,0,0)], [enc]), (64,64,64), (0,0,0))
    assert (vol == block).all()


@pytest.mark.parametrize('label', [0, 1, 2**64 - 1])
def test_roundtrip_uniform(label):
    block = np.full((64,64,64), label, np.uint64)
    enc = encode_label_block(block)

    # Uniform blocks consist of the header and the label only.
    assert len(enc) == 16 + 8
    vol = inflate_labelarray_blocks(_blocks_stream([(0,0,0)], [enc]), (64,64,64), (0,0,0))
    assert (vol == block).all()


def test_roundtrip_all_unique():
    # Worst case: every voxel is unique.
    block = np.arange(64**3, dtype=np.uint64).reshape((64,64,64))
    enc = encode_label_block(block)
    assert len(enc) == 16 + 8*64**3 + 2*512 + 4*64**3 + (9*512//8)*512
    vol = inflate_labelarray_blocks(_blocks_stream([(0,0,0)], [enc]), (64,64,64), (0,0,0))
    assert (vol == block).all()


def test_encode_noncontiguous():
    vol = _random_block(5).transpose()
    block = vol[:, :, :]
    assert not block.flags['C_CONTIGUOUS']
    enc = encode_label_block(block)
    inflated = inflate_labelarray_blocks(_blocks_stream([(0,0,0)], [enc]), (64,64,64), (0,0,0))
    assert (inflated == block).all()


def test_encode_bad_shape():
    with pytest.raises(ValueError):
        encode_label_block(np.zeros((64,64,32), np.uint64))


def test_inflate_volume():
    """
    Inflate several blocks into a volume, with a nonzero offset and a missing block.
    """
    offset = np.array((640, 1280, 1920))
    corners = [(0,0,0), (0,64,0), (64,0,0), (64,64,0)]
    vol = np.zeros((128,128,64), np.uint64)
    for (z, y, x), n in zip(corners, (1, 3, 17, 129)):
        vol[z:z+64, y:y+64, x:x+64] = _random_block(n, seed=n)

    encoded = [encode_label_block(vol[z:z+64, y:y+64, x:x+64]) for z, y, x in corners]

    # Omit the last block; it should be filled with zeros.
    stream = _blocks_stream(offset + corners[:3], encoded[:3])
    inflated = inflate_labelarray_blocks(stream, (128,128,64), offset)

    expected = vol.copy()
    expected[64:, 64:, :] = 0
    assert (inflated == expected).all()


def test_inflate_empty_stream():
    vol = inflate_labelarray_blocks(b'', (64,128,64), (0,0,0))
    assert vol.shape == (64,128,64)
    assert (vol == 0).all()


def test_inflate_block_outside_volume():
    enc = encode_label_block(_random_block(3))
    stream = _blocks_stream([(64,0,0)], [enc])
    with pytest.raises(ValueError):
        inflate_labelarray_blocks(stream, (64,64,64), (0,0,0))


def test_inflate_truncated_stream():
    enc = encode_label_block(_random_block(3))
    stream = _blocks_stream([(0,0,0)], [enc])
    with pytest.raises(ValueError):
        inflate_labelarray_blocks(stream[:-10], (64,64,64), (0,0,0))


def test_inflate_truncated_block():
    enc = encode_label_block(_random_block(3))
    stream = _blocks_stream([(0,0,0)], [enc[:-10]])
    with pytest.raises(ValueError):
        inflate_labelarray_blocks(stream, (64,64,64), (0,0,0))


# Generated with libdvid-cpp:
#   zlib.compress(libdvid.encode_label_block(_fixture_block()), 9)
LIBDVID_ENCODED_FIXTURE = """
eNrt2gdQlGmegPEWERERARERUckgggRJIkgUkJxzlGRARUUwYBpzzpgVkKCOOkZMmMbsGMfsqIPZ
cdTRGdO4jnP+6x6v9rbqbmtr72prS/rrrt/bVR2+r7rr/fp9qlUVCoUqN4WS4r8uTVEVNbAlamIj
/PzUxqiMzVANm6M6tkAt1MZWqIOtURfboB62RX1shwbYHjtgRzREIzRGEzRFMzRHC7TETmiFndEa
bbALNkEVtEU7tEcH7IqO6ITO6IKu2A3dsDu6owf2QE/0Qm/0QV/0w57ojwEYiL0wCIMxBEMxDMMx
AiMxCqMxBmMxDuMxARMxCZMxBVMxDdMxAzOxN2ZhNuZgLuZhH+yL/bA/5uMAHIiDsAAH4xAcioU4
DIuwGIfjCByJo7AER+MYHIvjcDx+hRNwIk7CyTgFp+I0nI4zcCbOwtk4B+fiPJyPC3AhLsJSXIxL
cCkuw+W4AlfiKlyNZViOFbgGK7EKq7EG1+I6XI9f4wbciJvwG9yMW3ArbsPtuANrcSfuwt24B/di
He7D/XgAD+Ih/BYP4xE8isfwOJ7Ak3gKv8PTeAbP4jk8jxfwe7yIl/AyXsGreA2v4w38AW/iLbyN
P2I93sG7eA/v4wN8iI/wMf6ET/BnfIrP8Dn+gi/wJf6Kv+ErfI1v8C2+w9/xPf4FP+Af+BH/VPz3
E3ojVMLGqIxNUAWboio2QzVsjurYAjWwJWqiFmpjK9TB1qiLbVAP26I+tkMDbI8dsCMaohEaowma
ohmaowVaYie0ws5ojTbYBW3RDu3RAbuiIzqhM7qgK3ZDN+yO7uiBPdATvdAbfdAX/bAn+mMABmIv
DMJgDMFQDMNwjMBIjMJojMFYjMN4TMBETMJkTMFUTMN0zMBM7I1ZmI05mIt52Af7Yj/sj/k4AAfi
ICzAwTgEh2IhDsMiLMbhOAJH4igswdE4BsfiOByPX+EEnIiTcDJOwak4DafjDJyJs3A2zsG5OA/n
4wJciIuwFBfjElyKy3A5rsCVuApXYxmWYwWuwUqswmqswbW4Dtfj17gBN+Im/AY34xbcittwO+7A
WtyJu3A37sG9WIf7cD8ewIN4CL/Fw3gEj+IxPI4n8CSewu/wNJ7Bs3gOz+MF/B4v4iW8jFfwKl7D
63gDf8CbeAtv449Yj3fwLt7D+/gAH+IjfIw/4RP8GZ/iM3yOv+ALfIm/4m/4Cl/jG3yL7/B3fI9/
wQ/4B37EPz+f9z+t5Bt/Wp1qfVoRe39aBXz6HfDpJ0DD1rA1bF/C9nkJoETLU/6bVtWUJtjsrx6n
RtdTp+dp/M1zW9ILteh7reh6rf+Bx+nS/PRoffo0PgPaXgeaniEtz5iGZ0q7M6fZWdLqrGh01v/D
sdnQ6v4vjvH/c99taYb2tMKuNEIn2qALTbAbLbA7DdCD9udJ8/tSjtObpulLy+xJwwygXfaiWQbT
KkNplOG0yUiaZDQtMpYGGU97TKQ5JtMaU2mM6bTFTJpiFi0xh4aYRzvsSzPsTyscQCMcRBscTBMc
SgscRgMspv2NoPmNovWNpvGNpe2Np+lNoOVNouFNaficv5jPeSp9djpddiY9djYddi79dT7ddSG9
tZTOuoS+uoyuuoKeuoqOWkY/raCbVtJLq+mka+mj6+miG+ihm+igm+mfW+me2+mdtXTOXfTNPXTN
OnrmfjrmQfrlt3TLI/TKY3TKE/TJU3TJ0/TIs3TI8/TH7+mOl+iNV+iM1+iLN+iKN+mJt+mI9fTD
u3TD+/TCh3TCx/TBJ3TBp/TA53TAF/S/X+l+r+h9b+h87+h77+l6H+h5Hz93vEb/2e+k3Um3k2Yn
vU5anXQ6aXTS56TNSZeTJic97nOL+9LPhQ1zR8Pc0TB3/ONzhzR86ffS7qXbS7OXXi+tXjq9NHrp
89LmpctLk5ceLy1eOrw0eOnv0t6lu0tzl94urV06uzR26evS1qWrS1OXni4tXTq6NHTp59LOpZtL
M5deLq1cOrk0cunj0sali0sTlx4uLVw6uDRw6d/SvqV7S/OW3i2tWzq3NG7p29K2pWtL05aeLS1b
OrY0bOnX0q6lW0uzll4trVo6tTRq6dPSpqVLS5OWHi0tWjq0NGjpz9KepTtLc5beLK1ZOrM0ZunL
0palK0tTlp4sLVk6sjRk6cfSjqUbSzOWXiytWDqxNGLpw9KGpQtLE5YeLC1YOrA0YOm/0n6l+0rz
ld4rrVc6rzRe6bvSdqXrStOVnistVzquNFzpt9JupdtKs5VeK61WOq00Wumz0maly0qTlR4rLVY6
rDRY6a/SXqW7SnOV3iqtVTqrNFbpq9JWpatKU5WeKi1VOqo0VOmn0k6lm0ozlV4qrVQ6qTRS6aPS
RqWLShOVHiotVDqoNFDpn9I+pXtK85TeKa1TOqc0Tumb0jala0rTlJ4pLVM6pjRM6ZfSLqVbSrOU
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
SlPU
"""


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_blockcodec'])
