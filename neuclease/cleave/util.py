import getpass
from io import BytesIO

import numpy as np
import pandas as pd
import requests


def fetch_body_edge_table(cleave_server, dvid_server, uuid, instance, body, timeout=60.0):
    """
    Note:
        If you give a body that doesn't exist, the server returns a 404 error.
    """
    dvid_server, dvid_port = dvid_server.split(':')

    if not cleave_server.startswith('http'):
        cleave_server = 'http://' + cleave_server

    data = { "body-id": body,
             "port": dvid_port,
             "server": dvid_server,
             "uuid": uuid,
             "segmentation-instance": instance,
             "user": getpass.getuser() }

    r = requests.post(f'{cleave_server}/body-edge-table', json=data, timeout=timeout)
    r.raise_for_status()

    df = pd.read_csv(BytesIO(r.content), header=0)
    df = df.astype({'id_a': np.uint64, 'id_b': np.uint64, 'score': np.float32})
    return df
