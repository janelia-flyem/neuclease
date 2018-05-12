[![Travis CI Status](https://travis-ci.org/janelia-flyem/neuclease.svg?branch=master)](https://travis-ci.org/janelia-flyem/neuclease)

# Neuclease

Tools for computing interactive "cleaves" of agglomerated neuron fragments from a DVID server.

## Install

```bash
conda install -c flyem-forge -c conda-forge neuclease
```

## Server Launch Example

```bash
neuclease_cleave_server \
    --port 5555 \
    --merge-table /path/to/merge-table.npy \
    --primary-dvid-server emdata3:8900 \
    --primary-uuid 017a \
    --primary-labelmap-instance segmentation \
    --log-dir /path/to/logs
```

## Example Client Request

```python
import requests

data = { "user": "bergs",
         "body-id": 673509195,
         "port": 8900,
         "seeds": {"1": [675222237], "2": [1266560684], "3": [1142805921]},
         "server": "emdata3.int.janelia.org", # DVID server
         "uuid": "017a",
         "segmentation-instance": "segmentation",
         "mesh-instance": "segmentation_meshes_tars" }

r = requests.post('http://bergs-ws1.int.janelia.org:5556/compute-cleave', json=data)
print(r.json())

# { "assignments": {
#      "1": [675222237, 12345, ...],
#      "2": [1266560684, 23456, ...],
#      "3": [1142805921, ....] },
#   ...
# }
```

### Optional dependency

One step in the algorithm can be sped up slightly if you install `graph-tool`:

```
conda install -c flyem-forge graph-tool
```

(Otherwise, `networkx` is used. It doesn't matter much, except for very large cleaves.)
