import numpy as np
import pandas as pd

from ._labelmap import fetch_sparsevol


def pointcloud_sample_for_body(dvid_server, uuid, instance, body, sparsevol_scale, num_points, session=None):
    """
    Sample a point cloud from a sparsevol.

    Args:
        dvid_server:
            DVID server
        uuid:
            DVID UUID
        instance:
            DVID labelmap instance
        body:
            body ID
        sparsevol_scale:
            The downres level of the voxels from which the body's sparse mask will be fetched.
            This determines the set of voxels from which the point sample will be drawn.
            (Higher scales -> fewer voxels -> fewer/shorter RLEs -> faster results, albeit at less precision.)
        num_points:
            Number of points to sample

    Returns:
        DataFrame with columns 'zyx'.
        Note:
            We always return the data in scale-0 coordinate space,
            regardless of the ``sparsevol_scale`` used to obtain the neuron mask.
            For example, if ``sparsevol_scale=2``, then all returned coordinates will be multiples of 4.

    Example usage:

        .. code-block:: python

        import json
        from neuclease.misc.neuroglancer import annotation_layer_json

        SHADER = '''
            void main() {
                setPointMarkerSize(1.0);
                setPointMarkerColor(vec4(defaultColor(), 1.0));
                setPointMarkerBorderWidth(1.0);
                setPointMarkerBorderColor(defaultColor());
            }
        '''

        # Runtime: 2.5 seconds
        brain_seg = ('emdata6.int.janelia.org:9000', ':master', 'segmentation')
        pc = pointcloud_sample_for_body(*brain_seg, 85115, sparsevol_scale=2, num_points=1000)

        # Produce a neuroglancer layer definition (as JSON),
        # which can be pasted into a neuroglancer JSON state for visualization.
        layer_json = annotation_layer_json(pc, 'point-cloud', color='#ffffff', shader=SHADER)
        print(json.dumps(layer_json))
    """
    starts_zyx, lengths = fetch_sparsevol(
        dvid_server,
        uuid,
        instance,
        body,
        sparsevol_scale,
        format='rle',
        session=session
    )

    # TODO: Allow custom random seed
    chosen_rows = np.random.choice(
        len(starts_zyx),
        num_points,
        replace=True,
        p=lengths / lengths.sum()
    )
    points_zyx = starts_zyx[chosen_rows]
    points_zyx[:, 2] += np.random.randint(0, lengths[chosen_rows])
    points_zyx *= (2**sparsevol_scale)

    return pd.DataFrame(points_zyx, columns=[*'zyx'])
