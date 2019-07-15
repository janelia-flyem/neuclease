from io import StringIO

import numpy as np
import pandas as pd

def swc_to_dataframe(swc_text):
    """
    Convert the given SWC file text into a pandas DataFrame with columns:
    ['node', 'kind', 'x', 'y', 'z', 'radius', 'parent']
    """
    if isinstance(swc_text, bytes):
        swc_text = swc_text.decode('utf-8')

    lines = swc_text.split('\n')
    lines = [*filter(lambda l: len(l) and l[0] not in '# \t', lines)]
    swc_text = '\n'.join(lines)
    
    columns = ['node', 'kind', 'x', 'y', 'z', 'radius', 'parent']
    dtypes = { 'node': np.int32,
               'kind': np.uint8,
               'x': np.float32,
               'y': np.float32,
               'z': np.float32,
               'radius': np.float32,
               'parent': np.int32 }

    df = pd.read_csv(StringIO(swc_text), sep=' ', names=columns, dtype=dtypes)
    return df

