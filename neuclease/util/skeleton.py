from io import BytesIO, StringIO

import numpy as np
import pandas as pd


def swc_to_dataframe(swc_text, neuprint_colnames=True):
    """
    Convert the given SWC file text into a pandas DataFrame with columns:
    ['node', 'kind', 'x', 'y', 'z', 'radius', 'parent']
    """
    if isinstance(swc_text, bytes):
        swc_text = swc_text.decode('utf-8')

    # Drop comments
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

    if neuprint_colnames:
        df = df.rename(columns={
            'node': 'rowId',
            'parent': 'link'}).drop(columns=['kind'])

    return df


def normalize_skeleton(skeleton_df):
    """
    Ensure that a skeleton's nodes have contiguous IDs from 1..N,
    and that they are listed in sorted order.

    Works IN-PLACE.
    """
    # Remap vertices to range 1..N
    if (skeleton_df['node'] == np.arange(1, 1+len(skeleton_df))).all():
        return skeleton_df

    # Use 0 for virtual root instead of -1
    assert skeleton_df['node'].min() >= 1
    assert skeleton_df['parent'].min() == -1
    skeleton_df.loc[skeleton_df['parent'] == -1, 'parent'] = 0

    num_nodes = skeleton_df['node'].nunique()
    remap = np.zeros(skeleton_df['node'].max()+1, dtype=int)
    remap[np.sort(skeleton_df['node'].unique())] = np.arange(1, num_nodes+1)

    skeleton_df['node'] = remap[skeleton_df['node'].values]
    skeleton_df['parent'] = remap[skeleton_df['parent'].values]
    skeleton_df = skeleton_df.sort_values('node', ignore_index=True)

    # Virtual root is -1 again
    skeleton_df.loc[skeleton_df['parent'] == 0, 'parent'] = -1

    assert (skeleton_df['node'] == np.arange(1, 1+len(skeleton_df))).all()
    assert skeleton_df['parent'].min() == -1
    assert not (skeleton_df['parent'] == 0).any()


def skeleton_to_neuroglancer(skeleton_df, orig_resolution_nm=8, output_path=None):
    """
    Convert a skeleton from DVID into the binary format that neuroglancer expects,
    as described here:
    https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/skeletons.md#encoded-skeleton-file-format
    """
    normalize_skeleton(skeleton_df)
    num_vertices = len(skeleton_df)
    num_edges = skeleton_df.eval('parent != -1').sum()
    vertex_positions = skeleton_df[[*'xyz']].values.astype(np.float32, 'C')
    vertex_positions[:] *= orig_resolution_nm
    edges = skeleton_df.query('parent != -1')[['node', 'parent']].values.astype(np.uint32, 'C')

    # Neuroglancer expects 0-based IDs
    edges -= 1

    stream = BytesIO()
    stream.write(np.uint32(num_vertices).tobytes())
    stream.write(np.uint32(num_edges).tobytes())
    stream.write(vertex_positions)
    stream.write(edges)

    b = stream.getvalue()
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(b)
    return b


def neuroglancer_skeleton_to_df(buf):
    """
    Converts a neuroglancer skeleton to a DataFrame.
    Leaves positions in nm units.
    Does not correspond to the original SWC from DVID.
    """
    if isinstance(buf, str):
        buf = open(buf, 'rb').read()

    stream = BytesIO(buf)
    num_vertices = np.frombuffer(stream.read(4), dtype=np.uint32)[0]
    num_edges = np.frombuffer(stream.read(4), dtype=np.uint32)[0]
    vertex_positions = np.frombuffer(stream.read(4*3*num_vertices), dtype=np.float32).reshape((-1, 3))
    edges = np.frombuffer(stream.read(4*2*num_edges), dtype=np.int32).reshape((-1, 2))

    if stream.tell() != len(buf):
        raise RuntimeError("Didn't consume the full buffer.  Skeleton is malformed or contains vertex attributes (unsupported)")

    # The neuroglancer spec doesn't specify a parent/child relationship,
    # so technically I should be doing some sort of toposort here,
    # but instead I'm going to assume that this skeleton was written
    # with the function above.
    edge_df = pd.DataFrame(edges, columns=['node', 'parent'])

    df = pd.DataFrame(vertex_positions, columns=[*'xyz'])
    df['node'] = np.arange(len(df), dtype=np.uint32)
    df = df.merge(edge_df, 'left', on='node')
    df['parent'].fillna(-1, inplace=True)
    df['parent'] = df['parent'].astype(np.int32)
    return df
