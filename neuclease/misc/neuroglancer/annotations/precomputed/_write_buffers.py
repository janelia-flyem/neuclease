import os
import shutil
from dataclasses import dataclass
from typing import Literal

from tqdm import tqdm
import tensorstore as ts

from ._util import _encode_uint64_series


def _write_buffers(buf_series, output_dir, subdir, write_sharded):
    """
    Write the buffers to the appropriate subdirectory of output_dir,
    in sharded or unsharded format.

    Args:
        buf_series:
            pd.Series of dtype=object, whose values are buffers (bytes objects).
            The index of the series provides the keys under which each item is stored.

        output_dir:
            str
            The directory into which the exported annotations will be written.

        subdir:
            str
            The subdirectory into which the buffers will be written.
            If it already exists, it will be deleted before we (re)create it.

        write_sharded:
            bool
            If True, write the buffers in sharded format.
            If False, write one file per item.

    Returns:
        JSON metadata for the written data, including the key (subdir)
        and sharding spec if applicable.
    """
    if os.path.exists(f"{output_dir}/{subdir}"):
        shutil.rmtree(f"{output_dir}/{subdir}")

    if write_sharded:
        return _write_buffers_sharded(buf_series, output_dir, subdir)
    else:
        return _write_buffers_unsharded(buf_series, output_dir, subdir)


def _write_buffers_unsharded(buf_series, output_dir, subdir):
    """
    Write the buffers to the appropriate subdirectory of output_dir,
    in unsharded format, i.e. one file per item.

    The index of buf_series is used as the key for each item, after being
    converted to a string (as decimal values in the case of integer keys).

    Returns:
        JSON metadata, always {"key": subdir}
    """
    # In the unsharded format, the keys are just strings (e.g. decimal IDs).
    string_keys = buf_series.index.astype(str)
    buf_series = buf_series.set_axis(string_keys)

    # Since we're writing unsharded files, we could have just used
    # standard Python open() and write() here for each key.
    # Using tensorstore here is mostly just a matter of taste, but it will
    # become useful if we ever support alternative storage backends such as gcs.
    kvstore = ts.KvStore.open(f"file://{output_dir}/{subdir}/").result()

    # Using a transaction here is not necessary, at least for plain files.
    # I'm not sure if it helps or hurts, but it probably doesn't matter much
    # for small datasets, which is presumably what we're dealing with if the
    # user has chosen the unsharded format.
    with ts.Transaction() as txn:
        for segment_key, buf in tqdm(buf_series.items(), total=len(buf_series)):
            kvstore.with_transaction(txn)[segment_key] = buf

    metadata = {"key": subdir}
    return metadata


def _write_buffers_sharded(buf_series, output_dir, subdir):
    """
    Write the buffers to the appropriate subdirectory of output_dir,
    in sharded format.

    The index of buf_series is used as the key for each item,
    after being encoded as a bigendian uint64.

    Returns:
        JSON metadata, including the output "key" (subdir) and sharding spec.
    """
    # When writing sharded data, we must use encoded bigendian uint64 as the key.
    # https://github.com/google/neuroglancer/pull/522#issuecomment-1923137085
    bigendian_keys = _encode_uint64_series(buf_series.index, '>u8')
    buf_series = buf_series.set_axis(bigendian_keys)

    shard_spec = _choose_output_spec(
        total_count=len(buf_series),
        total_bytes=buf_series.map(len).sum(),  # fixme, might be slow
        hashtype='murmurhash3_x86_128',
        gzip_compress=True
    )
    spec = {
        "driver": "neuroglancer_uint64_sharded",
        "metadata": shard_spec.to_json(),
        "base": f"file://{output_dir}/{subdir}",
    }
    kvstore = ts.KvStore.open(spec).result()

    # Note:
    #   At the time of this writing, tensorstore uses a
    #   surprising amount of RAM to perform these writes.
    with ts.Transaction() as txn:
        for segment_key, buf in tqdm(buf_series.items(), total=len(buf_series)):
            kvstore.with_transaction(txn)[segment_key] = buf

    metadata = {
        "key": subdir,
        "sharding": shard_spec.to_json()
    }
    return metadata


@dataclass
class ShardSpec:
    """
    Copied from Forrest Collman's PR:
    https://github.com/google/neuroglancer/pull/522
    """
    type: str
    hash: Literal["murmurhash3_x86_128", "identity_hash"]
    preshift_bits: int
    shard_bits: int
    minishard_bits: int
    data_encoding: Literal["raw", "gzip"]
    minishard_index_encoding: Literal["raw", "gzip"]

    def to_json(self):
        return {
            "@type": self.type,
            "hash": self.hash,
            "preshift_bits": self.preshift_bits,
            "shard_bits": self.shard_bits,
            "minishard_bits": self.minishard_bits,
            "data_encoding": str(self.data_encoding),
            "minishard_index_encoding": str(self.minishard_index_encoding),
        }


def _choose_output_spec(
    total_count,
    total_bytes,
    hashtype: Literal["murmurhash3_x86_128", "identity_hash"] = "murmurhash3_x86_128",
    gzip_compress=True,
):
    """
    Copied from Forrest Collman's PR:
    https://github.com/google/neuroglancer/pull/522
    """
    import tensorstore as ts
    MINISHARD_TARGET_COUNT = 1000
    SHARD_TARGET_SIZE = 50000000

    # if total_count == 1:
    #     return None
    # if ts is None:
    #     return None

    # test if hashtype is valid
    if hashtype not in ["murmurhash3_x86_128", "identity_hash"]:
        raise ValueError(
            f"Invalid hashtype {hashtype}."
            "Must be one of 'murmurhash3_x86_128' "
            "or 'identity_hash'"
        )

    total_minishard_bits = 0
    while (total_count >> total_minishard_bits) > MINISHARD_TARGET_COUNT:
        total_minishard_bits += 1

    shard_bits = 0
    while (total_bytes >> shard_bits) > SHARD_TARGET_SIZE:
        shard_bits += 1

    preshift_bits = 0
    while MINISHARD_TARGET_COUNT >> preshift_bits:
        preshift_bits += 1

    minishard_bits = total_minishard_bits - min(total_minishard_bits, shard_bits)
    data_encoding: Literal["raw", "gzip"] = "raw"
    minishard_index_encoding: Literal["raw", "gzip"] = "raw"

    if gzip_compress:
        data_encoding = "gzip"
        minishard_index_encoding = "gzip"

    return ShardSpec(
        type="neuroglancer_uint64_sharded_v1",
        hash=hashtype,
        preshift_bits=preshift_bits,
        shard_bits=shard_bits,
        minishard_bits=minishard_bits,
        data_encoding=data_encoding,
        minishard_index_encoding=minishard_index_encoding,
    )
