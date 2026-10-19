"""
Pure-Python (NumPy + pandas) replacement for dvidutils.LabelMapper.

    from labelmapper import LabelMapper

    mapper = LabelMapper(domain, codomain)      # build the hash table once
    out = mapper.apply(volume)                  # new array, codomain dtype
    out = mapper.apply(volume, allow_unmapped=True)
    out = mapper.apply_with_default(volume, default=0)
    mapper.apply_inplace(volume)                # overwrite volume

Same API and semantics as dvidutils.LabelMapper:
  - domain and codomain are 1-D arrays of equal length.
  - apply() returns a new array with the codomain's dtype.
  - apply_inplace() writes into the input array, casting mapped values to
    the input's dtype (like the C++ static_cast).
  - Unmapped labels raise an error unless allow_unmapped=True, in which case
    they pass through unchanged (cast to the output dtype).
  - If a key appears more than once in the domain, the last value wins.
  - The missing-label error is LabelMapperKeyError, which subclasses both
    KeyError and RuntimeError, so existing `except RuntimeError` code still
    catches it.

How it works, per call:
  1. The input is processed in slabs along its slowest-varying axis. The
     temporaries (int64 codes plus pandas' factorize hash table, which is
     sized to its input) cost about 16 bytes per voxel of a slab, so they're
     bounded by `slab_voxels` (default 2**22 voxels, about 64 MiB) instead
     of growing with the input. Measured on a 256^3 uint64 volume, this
     halved peak memory at a cost of a few percent in speed.
  2. Each slab is factorized with pandas: every voxel is hashed once, giving
     the slab's unique labels and an index into them.
  3. Only those unique labels are looked up in the persistent pd.Index.
  4. A single gather writes the mapped values straight into the output.

Memory and dtype handling:
  - Keys are stored in their own dtype (a uint32 domain gets a uint32 hash
    table; it is never widened to uint64).
  - Lookups are always cast to the key dtype before searching the index.
    pandas would otherwise cast the whole index to the wider dtype and
    rebuild its hash table on every call. Labels that don't fit in the key
    dtype can't be in the mapping, so they're treated as unmapped.
  - Values keep the codomain's dtype; no intermediate 64-bit output array
    is made.
"""
import numpy as np
import pandas as pd

__all__ = ["LabelMapper", "LabelMapperKeyError"]

DEFAULT_SLAB_VOXELS = 2**22


class LabelMapperKeyError(KeyError, RuntimeError):
    """Raised when a label is missing from the mapping and unmapped labels aren't allowed."""

    def __str__(self):
        # KeyError's default __str__ wraps the message in quotes.
        return str(self.args[0]) if self.args else ""


class LabelMapper:
    """
    Maps arrays of labels (the domain) to new labels (the codomain).

    Parameters
    ----------
    domain, codomain : 1-D integer arrays of equal length.
    slab_voxels : int or None, keyword-only
        Maximum number of voxels processed at once. Temporary memory is
        about 16 bytes per voxel of a slab (plus a copy of the slab if the
        input isn't contiguous). Slabs are whole rows along the slowest
        axis, so a slab can exceed this if a single row does. None
        processes the whole array in one pass.
    """

    def __init__(self, domain, codomain, *, slab_voxels=DEFAULT_SLAB_VOXELS):
        domain = np.asarray(domain)
        codomain = np.asarray(codomain)
        if domain.ndim != 1 or codomain.ndim != 1:
            raise ValueError("Can't initialize LabelMapper: domain and codomain should be 1D arrays")
        if domain.shape[0] != codomain.shape[0]:
            raise ValueError("Can't initialize LabelMapper: "
                             "domain and codomain arrays don't have matching sizes.")
        for name, a in (("domain", domain), ("codomain", codomain)):
            if not (np.issubdtype(a.dtype, np.integer) or len(a) == 0):
                raise TypeError(f"LabelMapper {name} must be an integer array, not {a.dtype}")
        if domain.dtype.kind not in "iu":
            domain = domain.astype(np.uint64)
        if codomain.dtype.kind not in "iu":
            codomain = codomain.astype(np.uint64)
        if slab_voxels is not None and slab_voxels < 1:
            raise ValueError("slab_voxels must be positive or None")

        index = pd.Index(domain, dtype=domain.dtype, copy=False)
        if not index.is_unique:
            # Match the C++ loop `_mapping[domain(i)] = codomain(i)`: last one wins.
            keep = ~index.duplicated(keep="last")
            domain, codomain = domain[keep], codomain[keep]
            index = pd.Index(domain, dtype=domain.dtype, copy=False)

        self._domain = np.ascontiguousarray(domain)
        self._codomain = np.ascontiguousarray(codomain)
        self._index = index
        self.slab_voxels = slab_voxels
        self._build_table()

    # -- public API (matches dvidutils.LabelMapper) --------------------------

    def apply(self, src, allow_unmapped=False):
        """Return a new array with each label mapped; dtype is the codomain's."""
        src = np.asarray(src)
        return self._apply(src, self._codomain.dtype, allow_unmapped=allow_unmapped)

    def apply_with_default(self, src, default=0):
        """Return a new array with each label mapped, and unmapped labels set to `default`."""
        src = np.asarray(src)
        return self._apply(src, self._codomain.dtype, allow_unmapped=True, default=default)

    def apply_inplace(self, src, allow_unmapped=False):
        """Map the labels of `src` in place. Mapped values are cast to src's dtype."""
        if not isinstance(src, np.ndarray):
            raise TypeError("apply_inplace() requires a numpy array")
        if not src.flags.writeable:
            raise ValueError("apply_inplace() requires a writeable array")
        self._apply(src, src.dtype, allow_unmapped=allow_unmapped, out=src)

    # -- extras ---------------------------------------------------------------

    def __len__(self):
        return len(self._domain)

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    def __getstate__(self):
        return {"domain": self._domain, "codomain": self._codomain, "slab_voxels": self.slab_voxels}

    def __setstate__(self, state):
        # Rebuild eagerly after unpickling, so threads never race to build the table.
        self._domain = state["domain"]
        self._codomain = state["codomain"]
        self.slab_voxels = state["slab_voxels"]
        self._index = pd.Index(self._domain, dtype=self._domain.dtype, copy=False)
        self._build_table()

    def __repr__(self):
        return (f"LabelMapper({len(self):,} entries, "
                f"{self._domain.dtype} -> {self._codomain.dtype}, slab_voxels={self.slab_voxels})")

    # -- implementation ---------------------------------------------------------

    def _build_table(self):
        # pd.Index builds its hash table lazily; force it now so the cost is
        # paid in the constructor, not the first apply().
        if len(self._domain):
            self._index.get_indexer(self._domain[:1])

    def _apply(self, src, out_dtype, allow_unmapped, default=None, out=None):
        if src.dtype.kind not in "iu":
            raise TypeError(f"LabelMapper can only map integer arrays, not {src.dtype}")

        # Process slabs along the slowest-varying axis, and give the output the
        # same memory layout as the input.
        if src.ndim > 1 and src.flags.f_contiguous and not src.flags.c_contiguous:
            order, axis = "F", src.ndim - 1
        else:
            order, axis = "C", 0

        if out is None:
            out = np.empty(src.shape, dtype=out_dtype, order=order)
        if src.size == 0:
            return out

        src_slab_view = np.moveaxis(src, axis, 0) if src.ndim else src.reshape(1)
        out_slab_view = np.moveaxis(out, axis, 0) if out.ndim else out.reshape(1)
        n_rows = src_slab_view.shape[0]
        row_voxels = max(1, src.size // n_rows)
        if self.slab_voxels is None:
            rows_per_slab = n_rows
        else:
            rows_per_slab = max(1, self.slab_voxels // row_voxels)

        for start in range(0, n_rows, rows_per_slab):
            stop = min(start + rows_per_slab, n_rows)
            s = np.moveaxis(src_slab_view[start:stop], 0, axis) if src.ndim else src_slab_view
            o = np.moveaxis(out_slab_view[start:stop], 0, axis) if out.ndim else out_slab_view
            self._apply_slab(s, o, order, out_dtype, allow_unmapped, default)
        return out

    def _apply_slab(self, src, out, order, out_dtype, allow_unmapped, default):
        flat = src.ravel(order=order)  # a view if the slab is contiguous
        codes, uniques = pd.factorize(flat, sort=False)
        mapped = self._map_uniques(uniques, out_dtype, allow_unmapped, default)

        out_is_contiguous = out.flags.c_contiguous if order == "C" else out.flags.f_contiguous
        if out_is_contiguous:
            # Gather straight into the output, no temporary. mode="clip" matters:
            # with the default mode="raise", np.take buffers `out` into a hidden
            # temporary (an extra full copy, and ~2.5x slower). Codes from
            # factorize are always valid indices, so clipping never changes them.
            np.take(mapped, codes, out=out.reshape(-1, order=order), mode="clip")
        else:
            out[...] = mapped[codes].reshape(out.shape, order=order)

    def _map_uniques(self, uniques, out_dtype, allow_unmapped, default):
        uniques = np.asarray(uniques)
        key_dtype = self._domain.dtype

        # Cast lookups to the key dtype (never the other way round: that
        # would make pandas rebuild the whole index). Labels outside the key
        # dtype's range can't be mapped.
        if uniques.dtype == key_dtype:
            pos = self._index.get_indexer(uniques)
        else:
            fits = _fits_mask(uniques, key_dtype)
            pos = np.full(len(uniques), -1, dtype=np.intp)
            if fits.any():
                pos[fits] = self._index.get_indexer(uniques[fits].astype(key_dtype))

        missing = pos == -1
        if len(self._codomain):
            # pos == -1 picks the last value; those entries are overwritten below.
            mapped = self._codomain[pos].astype(out_dtype, copy=False)
        else:
            mapped = np.zeros(len(pos), dtype=out_dtype)
        if missing.any():
            if not allow_unmapped:
                label = uniques[missing][0]
                raise LabelMapperKeyError(f"Label not found in mapping: {label}")
            if default is None:
                # Pass through, with the same wraparound as a C++ static_cast.
                mapped[missing] = uniques[missing].astype(out_dtype)
            else:
                mapped[missing] = np.array(default).astype(out_dtype)
        return mapped


def _fits_mask(a, dtype):
    """Boolean mask of the elements of integer array `a` representable in `dtype`."""
    info, ainfo = np.iinfo(dtype), np.iinfo(a.dtype)
    mask = np.ones(a.shape, dtype=bool)
    # Each threshold is only used when it lies inside a's own range, so it's
    # representable in a's dtype and the comparison is exact.
    if ainfo.min < info.min:
        mask &= a >= a.dtype.type(info.min)
    if ainfo.max > info.max:
        mask &= a <= a.dtype.type(info.max)
    return mask
