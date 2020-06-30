[![Travis CI Status](https://travis-ci.org/janelia-flyem/neuclease.svg?branch=master)](https://travis-ci.org/janelia-flyem/neuclease)

# Neuclease

Miscellaneous Python utilities for interacting with DVID and assisting with proofreading infrastructure.

Most users will only be interested in the [`neuclease.dvid` subpackage][ndvid], which provides Python bindings for most of [DVID's REST API][dvid-docs].
Eventually, that subpackage will be migrated into its own Python package.

[ndvid]: https://github.com/janelia-flyem/neuclease/tree/master/neuclease/dvid
[dvid-docs]: https://hemibrain-dvid.janelia.org/api/help/

Unfortunately, there is no web-based documentation at the moment.  But each function has a docstring.  Try your Python interpreter's built-in help feature.

## Install

```bash
conda install -c flyem-forge -c conda-forge neuclease
```

## Example

```python
In [1]: from neuclease.dvid import *

In [2]: server = 'http://hemibrain-dvid.janelia.org:8000'

In [3]: find_master('http://hemibrain-dvid.janelia.org:8000')
Out[3]: '20631f94c3f446d7864bc55bf515706e'

In [4]: x, y, z = 17019, 21341, 20789

In [5]: # Note: neuclease uses Z,Y,X order for all image volumes, coordinates, and bounding-boxes.
   ...: fetch_label(server, '20631f', 'segmentation', (z,y,x))
Out[5]: 5812980291

In [6]: fetch_label(server, '20631f', 'segmentation', (z,y,x), supervoxels=True)
Out[6]: 1351718075

In [7]: box_zyx = [[20789, 21341, 17019], [20889, 21441, 17119]]
   ...: subvol = fetch_labelmap_voxels(server, '20631f', 'segmentation', box)
   ...: subvol.shape
Out[7]: (100, 100, 100)

In [8]: # Read the docstring
   ...: fetch_labelmap_voxels?
Signature:
fetch_labelmap_voxels(
    server,
    uuid,
    instance,
    box_zyx,
    scale=0,
    throttle=False,
    supervoxels=False,
    *,
    format='array',
    session=None,
)
Docstring:
Fetch a volume of voxels from the given instance.

Args:
    server:
        dvid server, e.g. 'emdata3:8900'

    uuid:
        dvid uuid, e.g. 'abc9'

    instance:
        dvid instance name, e.g. 'segmentation'

    box_zyx:
        The bounds of the volume to fetch in the coordinate system for the requested scale.
        Given as a pair of coordinates (start, stop), e.g. [(0,0,0), (10,20,30)], in Z,Y,X order.
        The box need not be block-aligned, but the request to DVID will be block aligned
        to 64px boundaries, and the retrieved volume will be truncated as needed before
        it is returned.

    scale:
        Which downsampling scale to fetch from

    throttle:
        If True, passed via the query string to DVID, in which case DVID might return a '503' error
        if the server is too busy to service the request.
        It is your responsibility to catch DVIDExceptions in that case.

    supervoxels:
        If True, request supervoxel data from the given labelmap instance.

    format:
        If 'array', inflate the compressed voxels from DVID and return an ordinary ndarray
        If 'lazy-array', return a callable proxy that stores the compressed data internally,
        and that will inflate the data when called.
        If 'raw-response', return DVID's raw /blocks response buffer without inflating it.

Returns:
    ndarray, with shape == (box[1] - box[0])
```
