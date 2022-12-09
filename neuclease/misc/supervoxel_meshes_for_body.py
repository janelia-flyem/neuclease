"""
Example utility for generating supervoxel meshes
suitable for the neu3 cleaving protocol.

See the source code for argument definitions.

Example:

    supervoxel_meshes_for_body emdata6.int.janelia.org:9000 df383 segmentation 32939 --scale=4 --smoothing=2 --format=drc

Note:
    This tool fetches the sparsevol (RLE) representation of the body.
    That's fine for generating the meshes for a single body,
    but if you're generating meshes for an entire volume,
    it could be somewhat inefficient. In that case, you'd effectively
    end up reading from the same blocks of segmentation over and over.
"""
import logging
logger = logging.getLogger(__name__)


def main():
    import os
    import argparse

    from neuclease import configure_default_logging
    configure_default_logging()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('server')
    parser.add_argument('uuid')
    parser.add_argument('instance')
    parser.add_argument('body', type=int)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--smoothing', type=int, default=0)
    parser.add_argument('--decimation', type=float, default=1.0)
    parser.add_argument('--format', choices=['obj', 'drc'], default='drc')
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--processes', type=int, default=8)
    parser.add_argument('--output-dir')
    args = parser.parse_args()

    sv_meshes = supervoxel_meshes_for_body(
        args.server,
        args.uuid,
        args.instance,
        args.body,
        scale=args.scale,
        smoothing=args.smoothing,
        decimation=args.decimation,
        format=args.format,
        upload=args.upload,
        processes=args.processes
    )

    d = args.output_dir or f'{args.body}-meshes'
    os.makedirs(d, exist_ok=True)
    logging.info(f"Writing supervoxel meshes to {d}/")
    for sv, mesh_bytes in sv_meshes.items():
        with open(f'{d}/{sv}.{args.format}', 'wb') as f:
            f.write(mesh_bytes)


def supervoxel_meshes_for_body(server, uuid, instance, body, scale=2,
                               block_shape=(64, 64, 64), halo=4, smoothing=0, decimation=1.0,
                               format='drc', upload=False, processes=8):
    """
    Generate all supervoxel meshes for a body.

    For most argument definitions, see supervoxel_mesh(), above.

    Args:
        server:
            dvid server, e.g. "myserver.mydomain.org:8000"
        uuid:
            dvid DAG node
        instance:
            a labelmap instance name, e.g. "segmentation"
        body:
            body ID
        scale:
            int, Which downres scale to use (e.g. 0,1,2, etc.)
        block_shape:
            The supervoxel will be processed in chunks of this size.
            Keep in mind that this is used regardless of the scale you chose,
            so if you increase the 'scale' arg without changing the block_shape,
            then the neuron will be divided into fewer, larger chunks.
        halo:
            To avoid visible gaps between mesh chunks,
            it's best to make the chunks overlap with a small halo.
            This specifies the halo width.
        smoothing:
            How many rounds of laplacian smoothing to apply to the mesh.
        decimation:
            Reduce vertex count by applying decimation.
            This is the fraction of voxels which should REMAIN after decimation.
            If decimation=1.0, then no decimation is performed at all.
        format:
            Which mesh format to generate,
            either 'drc' (draco), or 'obj' (Wavefront OBJ).
        upload:
            If True, upload the supervoxel meshes to a tarsupervoxels instance,
            named '{instance}_sv_meshes'
        processes:
            If non-zero, use a process pool to compute supervoxel meshes in parallel.

    Returns:
        dict of {sv_id: serialized_mesh_bytes}
    """
    from functools import partial
    from neuclease.util import compute_parallel
    from neuclease.dvid.tarsupervoxels import post_load
    from neuclease.dvid.labelmap import fetch_supervoxel_sizes_for_body

    sv_sizes = fetch_supervoxel_sizes_for_body(server, uuid, instance, body)

    # Little optimization:
    # When computing in parallel, it's optimal to get the big ones started first.
    # Otherwise, we might get unlucky and get a really big one last, leaving just one core occupied.
    sv_sizes = sv_sizes.sort_values(ascending=False)

    func = partial(
        supervoxel_mesh,
        server, uuid, instance,
        # sv,
        scale=scale, block_shape=block_shape, halo=halo,
        smoothing=smoothing, decimation=decimation,
        serialize=format
    )
    meshes = compute_parallel(func, sv_sizes.index, processes=processes)
    sv_meshes = dict(zip(sv_sizes.index, meshes))
    sv_meshes = {f'{sv}.drc': mesh for sv, mesh in sv_meshes.items()}

    if upload:
        post_load(server, uuid, f'{instance}_sv_meshes', sv_meshes)

    return sv_meshes


def supervoxel_mesh(server, uuid, instance, sv, scale=2,
                    block_shape=(64, 64, 64), halo=4, smoothing=0, decimation=1.0,
                    serialize=None):
    """
    Generate a mesh for a single supervoxel by fetching its sparsevol (RLE)
    representation, chunking that, computing a mesh for each chunk, and then
    naively combining the chunks.  No attempt is made to "stitch" the mesh chunks;
    they're just concatenated into a single file.  Hence, the mesh will not be
    "water tight", but that's often just fine for visualization.

    Args:
        server:
            dvid server, e.g. "myserver.mydomain.org:8000"
        uuid:
            dvid DAG node
        instance:
            a labelmap instance name, e.g. "segmentation"
        sv:
            supervoxel ID
        scale:
            int, Which downres scale to use (e.g. 0,1,2, etc.)
        block_shape:
            The supervoxel will be processed in chunks of this size.
            Keep in mind that this is used regardless of the scale you chose,
            so if you increase the 'scale' arg without changing the block_shape,
            then the neuron will be divided into fewer, larger chunks.
        halo:
            To avoid visible gaps between mesh chunks,
            it's best to make the chunks overlap with a small halo.
            This specifies the halo width.
        smoothing:
            How many rounds of laplacian smoothing to apply to the mesh.
        decimation:
            Reduce vertex count by applying decimation.
            This is the fraction of voxels which should REMAIN after decimation.
            If decimation=1.0, then no decimation is performed at all.
        serialize:
            If provided, then serialize the result to bytes in either 'drc' or 'obj' format.
            Otherwise, return a vol2mesh.Mesh object.

    Returns:
        Either vol2mesh.Mesh or bytes, depending
        on whether or not 'serialize' was given.
    """
    from neuclease.dvid.rle import blockwise_masks_from_ranges
    from neuclease.dvid.labelmap import fetch_sparsevol

    # Note: External dependency
    from vol2mesh import Mesh

    assert 0.0 < decimation <= 1.0
    ranges = fetch_sparsevol(server, uuid, instance, sv, scale, supervoxels=True, format='ranges')
    boxes, masks = blockwise_masks_from_ranges(ranges, block_shape, halo)

    block_meshes = []
    for box, mask in zip(boxes, masks):
        # box was specified in downsampled coordinates, but this
        # function will always return a mesh that uses scale-0 coordinates.
        box *= 2**scale
        m = Mesh.from_binary_vol(mask, box, method='ilastik', smoothing_rounds=smoothing)
        m.simplify(decimation)
        block_meshes.append(m)

    if block_meshes:
        mesh = Mesh.concatenate_meshes(block_meshes, keep_normals=False)
        if serialize:
            return mesh.serialize(fmt=serialize)
        else:
            return mesh

    if serialize:
        # Special protocol:
        # If a supervoxel is too small to generate a mesh for,
        # we store an empty file for it.
        # This is recognized in neu3 as "nothing to show, but not an error".
        return b''
    else:
        return None


if __name__ == "__main__":
    main()
