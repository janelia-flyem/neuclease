import os
import json
import zlib
from itertools import chain

import numpy as np
import pandas as pd

from ..util import tqdm_proxy
from ..dvid import fetch_mapping

EXAMPLE_ASSIGNMENT = {
  "file type":"Neu3 task list",
  "file version":1,
  "task list":[
    {
      "task type":"merge review",
      "task id":"0x16fa39e8",
      "supervoxel IDs":[
        2127892639,
        2127892711
      ],
      "supervoxel IDs 0.5":[
        2127892711
      ],
      "original_uuid":"54f7",
      "cc_index":116964,
      "original_bodies":[
        2127892639,
        2127892711
      ],
      "total_body_count":2
    },
  ]
}

def generate_mergereview_assignment(server, uuid, instance, sv_groups, focused_bodies, output_path=None):
    d = os.path.dirname(output_path)
    os.makedirs(d, exist_ok=True)
    
    focused_bodies = set(focused_bodies)
    all_svs = set(chain(*sv_groups.values()))
    sv_bodies = fetch_mapping(server, uuid, instance, all_svs)
    sv_bodies = pd.DataFrame(sv_bodies)
    all_05_svs = sv_bodies.query('body in @focused_bodies').index
    
    tasks = []
    for cc_index, svs in tqdm_proxy(sv_groups.items(), leave=False, disable=(len(sv_groups) < 1000)):
        svs = np.asarray(svs, order='C')
        svs_05 = all_05_svs.intersection(svs)

        task = {
            'task type': "merge review",
            'task id': hex(zlib.crc32(svs)),
            'original_uuid': uuid,
            'cc_index': int(cc_index),
            'supervoxel IDs': svs.tolist(),
            'supervoxel IDs 0.5': svs_05.tolist(),
            'total_body_count': len(pd.unique(sv_bodies.loc[svs, 'body']))
        }
        tasks.append(task)
    
    assignment = {
        "file type":"Neu3 task list",
        "file version":1,
        "task list ": tasks
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(assignment, f, indent=2)
    
    return assignment

def generate_mergereview_assignments(server, uuid, instance, sv_groups, focused_bodies, assignment_size, output_dir, prefix='assignment-'):
    sv_groups_list = list(sv_groups.items())
    for i, batch_start in enumerate(tqdm_proxy(range(0, len(sv_groups), assignment_size), leave=False)):
        path = f"{output_dir}/{prefix}{i:04d}.json"
        batch_sv_groups = dict(sv_groups_list[batch_start:batch_start+assignment_size])
        generate_mergereview_assignment(server, uuid, instance, batch_sv_groups, focused_bodies, path)
