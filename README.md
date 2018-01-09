This is a tiny server to wrap Jeremy Maitin-Shepard's greedy multicut solver for cleaving undersegmentations along supervoxel boundaries.

Prerequisites:

```
conda install python=2.7 numpy flask
pip install neuroglancer
```

Start Server:

```
$ python cleave_server.py --graph-db exported_merge_graphs/274750196357\:janelia-flyem-cx-flattened-tabs\:sec24_seg_v2a\:ffn_agglo_pass1_cpt5663627_medt160_with_celis_cx2-2048_r10_mask200_0.sqlite
```

Request a cleave:

```python

In [0]: import requests

In [1]: cleave_request_body = {
   ...:   "body-id": 107017171,
   ...:   "seeds": {
   ...:     "1": [107017171],
   ...:     "2": [116271579, 116271681]
   ...:   }
   ...: }

In [2]: r = requests.post('http://http://bergs-ws1.int.janelia.org:5555/compute-cleave', json=cleave_request_body)

In [3]: print(r.content.decode())
{
  "assignments": {
    "1": [
      107017171,
      119767911,
      119767929,
      116271630,
      119562242,
      119562117,
      110101985,
      119562125,
      119562235,
      119356299,
      119767898
    ],
    "2": [
      116271579,
      116271681,
      116271854,
      116271570,
      116271801,
      116477338,
      116477349,
      116271577,
      116271617
    ]
  },
  "body-id": 107017171
}
```
