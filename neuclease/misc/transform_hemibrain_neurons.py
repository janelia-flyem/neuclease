"""
Transform a list of hemibrain neurons (skeletons, meshes, or both)
to male CNS using John Bogovic's alignment scripts.
"""
import os
import sys
import glob
import logging
import argparse
import tempfile
import subprocess

logger = logging.getLogger(__name__)

Hemibrain_v12 = ('emdata4:8900', '31597d95bd844060b0ccc928a1a8a0a4')

# John Bogovic's programs to transform hemibrain points into unisex template and unisex to CNS
# Warning: Despite the name, this hemibrain script expects voxel units, not nm
# Expects voxels, produces microns
Hemibrain_to_unisex = '/nrs/saalfeld/john/flyem_maleBrain/transformation_2022Apr07/csv_hemibrainNm-JRC2018Uum'

# Expects um, produces nm
Unisex_to_cns = '/nrs/saalfeld/john/flyem_maleBrain/transformation_2022July11/csv_JRC2018Uum-to-EMnm'

UNISEX_MIDLINE_MICRONS = 313.5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--matching-only', action='store_true', help='Only process those bodies whose CNS match is listed in the CSV file')
    parser.add_argument('--target-space', '-t', choices=['unisex-template', 'male-cns'], default='male-cns')
    parser.add_argument('--info', action='store_true', help='Write (or overwrite) neuroglancer metadata')
    parser.add_argument('--skeleton', action='store_true', help='Download and transform skeletons')
    parser.add_argument('--mesh', action='store_true', help='Download and transform meshes')
    parser.add_argument('--reflect', action='store_true', default=False, help='Reflect the skeletons in template space')
    parser.add_argument('--skip-existing', action='store_true', help="Don't process bodies whose skeletons and/or meshes already exist in the output directory")
    parser.add_argument('--starting-index', type=int, default=1, help="Discard rows of the input CSV before this row (first row is 0)")
    parser.add_argument('--count', type=int, help="Process this many bodies, discarding all subsequent rows of the input CSV")
    parser.add_argument('--processes', '-p', type=int, default=0, help='How many processes to use when downloading skeletons/meshes')
    parser.add_argument('body_csv')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    from neuclease import configure_default_logging
    configure_default_logging()

    transform_hemibrain_neurons(args)
    logger.info("DONE")


def transform_hemibrain_neurons(args):
    if not args.mesh and not args.skeleton and not args.info:
        raise RuntimeError("Please specify at least one of --info --mesh --skeleton")

    body_df = _make_body_df(args)
    if len(body_df) == 0:
        sys.exit("No bodies to process")

    if args.info:
        write_neuroglancer_info(args, body_df)

    if not args.skeleton and not args.mesh:
        return

    body_df, hemi_df = _fetch_hemi_data(args, body_df)
    output_df = _transform_points(args, hemi_df)
    _write_files(args, body_df, output_df)


def _make_body_df(args):
    """
    Read the user's input CSV and processing options,
    and construct a DataFrame that outlines which
    bodies will be processed and what their final object IDs will be.
    """
    import pandas as pd
    from neuprint import Client, fetch_neurons

    body_df = pd.read_csv(args.body_csv)

    if len(body_df) == 0:
        sys.exit("Body list is empty")

    # Select a subset of the input (useful for processing array jobs on the cluster)
    start = args.starting_index
    count = args.count or len(body_df) - start
    body_df = body_df.iloc[start:start+count]

    if len(body_df) == 0:
        sys.exit("No bodies in specified subset")

    if args.matching_only:
        if args.reflect:
            body_df = body_df.query('not cns_body_counterpart.isnull()').copy()
        else:
            body_df = body_df.query('not cns_body.isnull()').copy()

    if len(body_df) == 0:
        sys.exit("No hemibrain bodies were listed with a matching CNS body")

    dupes = body_df.loc[body_df['hemibrain_body'].duplicated(), 'hemibrain_body']
    if len(dupes) > 0:
        logger.warning(
            "Some hemibrain bodies are listed multiple times. "
            f"Only the first will be processed: {sorted(dupes.unique().tolist())}")

    body_df = body_df.drop_duplicates('hemibrain_body')

    try:
        Client('neuprint.janelia.org', 'hemibrain:v1.2.1')
        neurons = fetch_neurons(body_df['hemibrain_body'].values)[0].set_index('bodyId').rename_axis('hemibrain_body')
    except Exception:
        # Try again in case of timeout
        # (If we run on the cluster, we might be overloading the server.)
        Client('neuprint.janelia.org', 'hemibrain:v1.2.1')
        neurons = fetch_neurons(body_df['hemibrain_body'].values)[0].set_index('bodyId').rename_axis('hemibrain_body')

    neurons['instance'] = neurons['instance'].fillna(neurons['type'])
    instances = neurons['instance']
    body_df = body_df.merge(instances, 'left', on='hemibrain_body')
    body_df['instance'] = body_df['instance'].fillna('') + ' (' + body_df['hemibrain_body'].astype(str) + ')'
    body_df['object_id'] = body_df['hemibrain_body']

    if 'cns_body_counterpart' in body_df.columns and args.reflect:
        body_df['object_id'] = body_df['cns_body_counterpart']
    elif 'cns_body' in body_df.columns:
        body_df['object_id'] = body_df['cns_body']
    body_df['object_id'] = body_df['object_id'].fillna(body_df['hemibrain_body']).astype(int)

    if args.skip_existing:
        existing_skeleton_files = glob.glob(f"{args.output_dir}/skeleton/*")
        existing_skeleton_files = (p.split('/')[-1] for p in existing_skeleton_files)
        existing_skeleton_files = filter(str.isnumeric, existing_skeleton_files)
        existing_skeleton_bodies = [*map(int, existing_skeleton_files)]  # noqa

        existing_mesh_files = glob.glob(f"{args.output_dir}/mesh/*.ngmesh")
        existing_mesh_files = (p.split('/')[-1][:-len(".ngmesh")] for p in existing_mesh_files)
        existing_mesh_bodies = [*map(int, existing_mesh_files)]  # noqa

        if args.skeleton and args.mesh:
            body_df = body_df.query('object_id not in @existing_skeleton_bodies or object_id not in @existing_mesh_bodies')
        elif args.skeleton:
            body_df = body_df.query('object_id not in @existing_skeleton_bodies')
        elif args.mesh:
            body_df = body_df.query('object_id not in @existing_mesh_bodies')

    if len(body_df) == 0:
        logger.info("All bodies already have existing skeleton and/or mesh files.")
        sys.exit(0)

    return body_df


def write_neuroglancer_info(args, body_df):
    """
    Write the neuroglancer 'info' files for a skeletons directory and a mesh directory.
    and also create a 'segment_properties' directory for each, using
    properties metadata from the data in body_df.

    Note:
        We always rewrite the info from scratch!
        Pre-existing info will be overwritten, and any
        properties on existing neurons will be deleted.
    """
    from neuclease.util import dump_json
    logger.info(f"Writing neuroglancer metadata to {args.output_dir}")

    props = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": [],
            "properties": [
                {
                    "id": "source",
                    "type": "label",
                    "values": []
                }
            ]
        }
    }

    body_df = body_df.sort_values('object_id')
    for object_id, instance in body_df[['object_id', 'instance']].values:
        props["inline"]["ids"].append(str(object_id))
        props["inline"]["properties"][0]["values"].append(instance)

    if args.skeleton or os.path.exists(f"{args.output_dir}/skeleton"):
        os.makedirs(f"{args.output_dir}/skeleton", exist_ok=True)
        dump_json(
            {
                "@type": "neuroglancer_skeletons",
                "segment_properties": "segment_properties"
            },
            f"{args.output_dir}/skeleton/info"
        )
        props_dir = f"{args.output_dir}/skeleton/segment_properties"
        os.makedirs(props_dir, exist_ok=True)
        dump_json(props, f"{props_dir}/info", unsplit_int_lists=True)

    if args.mesh or os.path.exists(f"{args.output_dir}/mesh"):
        os.makedirs(f"{args.output_dir}/mesh", exist_ok=True)
        dump_json(
            {
                "@type": "neuroglancer_legacy_mesh",
                "segment_properties": "segment_properties"
            },
            f"{args.output_dir}/mesh/info"
        )
        props_dir = f"{args.output_dir}/mesh/segment_properties"
        os.makedirs(props_dir, exist_ok=True)
        dump_json(props, f"{props_dir}/info", unsplit_int_lists=True)


def _fetch_hemi_data(args, body_df):
    """
    Fetch the skeletons and meshes for each of the bodies in body_df.
    Return the skeleton nodes and mesh vertices in one gigantic DataFrame,
    with extra columns indicating which body and source (skeleton vs. mesh)
    they came from. Also return Mesh objects as a new column in body_df.
    """
    import pandas as pd
    from neuclease.util import compute_parallel

    hemi_dfs = []
    if args.skeleton:
        logger.info(f"Fetching {len(body_df)} skeletons")
        skeletons = compute_parallel(_fetch_hemibrain_skeleton, body_df['hemibrain_body'], processes=args.processes)
        skeletons = [*filter(lambda x: x is not None, skeletons)]
        if len(skeletons) > 0:
            # Create a giant DataFrame of all skeleton points
            skeleton_df = pd.concat(skeletons, ignore_index=True)
            hemi_dfs.append(skeleton_df)

    if args.mesh:
        logger.info(f"Fetching {len(body_df)} meshes")
        meshes_and_dfs = compute_parallel(_fetch_hemibrain_mesh, body_df['hemibrain_body'], processes=args.processes)
        meshes_and_dfs = [*filter(None, meshes_and_dfs)]
        if len(meshes_and_dfs) == 0:
            body_df['mesh'] = None
        else:
            hemi_bodies, meshes, vertices_dfs = zip(*meshes_and_dfs)

            # Create a giant DataFrame of all Mesh vertices
            vertices_df = pd.concat(vertices_dfs, ignore_index=True)
            hemi_dfs.append(vertices_df)

            # Create a column in body_df for the Mesh objects
            mesh_df = pd.DataFrame({'hemibrain_body': hemi_bodies, 'mesh': meshes})
            body_df = body_df.merge(mesh_df, 'left', on='hemibrain_body')
            body_df.loc[body_df['mesh'].isnull(), 'mesh'] = None

    if len(hemi_dfs) == 0:
        sys.exit("None of the hemibrain objects could be fetched")

    hemi_df = pd.concat(hemi_dfs, ignore_index=True)
    return body_df, hemi_df


def _fetch_hemibrain_skeleton(hemi_body):
    """
    Fetch the skeleton for a hemibrain neuron,
    and attach columns for body and 'source'.
    """
    from requests import HTTPError
    from tqdm import tqdm
    from neuclease.dvid import fetch_skeleton

    try:
        df = fetch_skeleton(*Hemibrain_v12, 'segmentation_skeletons', hemi_body, 'pandas')
        df['hemibrain_body'] = hemi_body
        df['source'] = 'skeleton'
        return df
    except HTTPError:
        with tqdm.external_write_mode():
            logger.error(f"Failed to fetch skeleton for body {hemi_body}")
        return None


def _fetch_hemibrain_mesh(hemi_body):
    """
    Fetch the 'ngmesh' (single-resolution neuroglancer legacy mesh)
    for a given neuron. Returns the Mesh object, along with the mesh
    vertices extracted in a separate DataFrame.
    """
    import pandas as pd
    from requests import HTTPError
    from tqdm import tqdm
    from neuclease.dvid import fetch_key
    from vol2mesh import Mesh

    try:
        buf = fetch_key(*Hemibrain_v12, 'segmentation_meshes', f'{hemi_body}.ngmesh')
        m = Mesh.from_buffer(buf, fmt='ngmesh')

        # Convert from nm to voxels
        m.vertices_zyx = m.vertices_zyx / 8

        df = pd.DataFrame(m.vertices_zyx, columns=[*'zyx'])
        df['hemibrain_body'] = hemi_body
        df['source'] = 'mesh'
        return hemi_body, m, df
    except HTTPError:
        with tqdm.external_write_mode():
            logger.error(f"Failed to fetch mesh for body {hemi_body}")
        return None


def _transform_points(args, hemi_df):
    """
    Transform all coordinates in the given DataFrame into the target
    coordinate space (either unisex-template or male-cns).
    """
    # Perform conversion all in one step,
    # (The computation is dominated by overhead, so parallelism doesn't help much here,
    # and apparently the conversion program isn't multiprocess-safe.)
    from neuclease.util import Timer
    with Timer(f"Transforming {len(hemi_df)} points to unisex space", logger):
        unisex_df = transform_hemibrain_points_to_unisex(hemi_df)

    if args.reflect:
        logger.info(f"Reflecting {len(hemi_df)} points across the unisex midline")
        midline = UNISEX_MIDLINE_MICRONS * 1000 / 8  # noqa
        unisex_df['x'] = (unisex_df.eval('2 * @midline - x'))

    assert args.target_space in ('male-cns', 'unisex-template')
    if args.target_space == 'unisex-template':
        return unisex_df

    with Timer(f"Transforming {len(unisex_df)} points to CNS space", logger):
        return transform_unisex_points_to_cns(unisex_df)


def transform_hemibrain_points_to_unisex(hemi_df, tmpdir=None):
    """
    Transform a hemibrain neuron to unisex template space.
    The resulting skeleton is in voxel units, per FlyEM/DVID conventions.
    """
    import numpy as np
    import pandas as pd

    if not tmpdir:
        tmpdir = tempfile.mkdtemp()
    hemi_csv = f'{tmpdir}/hemi.csv'
    unisex_csv = f'{tmpdir}/unisex.csv'

    # Note: The hemibrain script expects voxels, not nm
    hemi_df[[*'xyz']].to_csv(hemi_csv, header=False, index=False)

    # Run John's conversion
    # Note: result is in microns, not nm or voxels.
    try:
        subprocess.run([Hemibrain_to_unisex, hemi_csv, unisex_csv], check=True, capture_output=True)
    except subprocess.CalledProcessError as ex:
        print(ex.stdout.decode('utf-8'), '\n', ex.stderr.decode('utf-8'), file=sys.stderr)
        raise

    unisex_coords = pd.read_csv(unisex_csv, header=None, names=[*'xyz'])

    # Overwrite with transformed points, converted to voxels
    unisex_df = hemi_df.copy()
    unisex_df[[*'xyz']] = unisex_coords * 1000 / 8
    unisex_df[[*'xyz']] = unisex_df[[*'xyz']].astype(np.float32)
    return unisex_df


def transform_unisex_points_to_cns(unisex_df, tmpdir=None):
    """
    Transform a unisex neuron to CNS template space.
    We assume the input is in voxel units, per FlyEM/DVID conventions,
    and the result will also be in voxel units.
    """
    import numpy as np
    import pandas as pd

    if not tmpdir:
        tmpdir = tempfile.mkdtemp()
    unisex_csv = f'{tmpdir}/unisex.csv'
    cns_csv = f'{tmpdir}/cns.csv'

    # Convert from voxels to to microns
    unisex_df = unisex_df.copy(deep=True)
    unisex_df[[*'xyz']] *= 8/1000
    unisex_df[[*'xyz']].to_csv(unisex_csv, index=False, header=False)

    # Run John's conversion
    # Note: result is in nm, not voxels or microns
    try:
        subprocess.run([Unisex_to_cns, unisex_csv, cns_csv], check=True, capture_output=True)
    except subprocess.CalledProcessError as ex:
        print(ex.stdout.decode('utf-8'), '\n', ex.stderr.decode('utf-8'), file=sys.stderr)
        raise

    cns_coords = pd.read_csv(cns_csv, header=None, names=[*'xyz'])

    # Overwrite with transformed points, converted from nm to voxels
    cns_df = unisex_df.copy()
    cns_df[[*'xyz']] = cns_coords / 8
    cns_df[[*'xyz']] = cns_df[[*'xyz']].astype(np.float32)
    return cns_df


def _write_files(args, body_df, output_df):
    """
    Write skeleton and mesh files in neuroglancer format,
    using the coordinates given in output_df and the Mesh
    faces from the 'mesh' column in body_df.
    """
    from neuclease.util import skeleton_to_neuroglancer
    body_df = body_df.set_index('hemibrain_body')

    if args.skeleton:
        os.makedirs(f"{args.output_dir}/skeleton", exist_ok=True)
    if args.mesh:
        os.makedirs(f"{args.output_dir}/mesh", exist_ok=True)

    for (source, hemi_body), df in output_df.groupby(['source', 'hemibrain_body'], sort=False):
        assert source in ('skeleton', 'mesh')
        object_id = body_df.loc[hemi_body, 'object_id']
        if source == 'skeleton':
            try:
                skeleton_to_neuroglancer(df, 8, f"{args.output_dir}/skeleton/{object_id}")
            except Exception as ex:
                logger.error(f"Failed to write skeleton for hemibrain body {hemi_body}: {ex}")
        if source == 'mesh':
            mesh = body_df.loc[hemi_body, 'mesh']
            if mesh:
                mesh_to_neuroglancer(object_id, df, mesh, 8, args.output_dir)


def mesh_to_neuroglancer(object_id, vertices_df, mesh, resolution, output_dir):
    """
    Convert a mesh to neuroglancer format, overwriting the vertices first.
    """
    from neuclease.util import dump_json

    # Overwrite with transformed points, and convert to nm
    mesh.vertices_zyx = resolution * vertices_df[[*'zyx']].values

    # Dump mesh file and fragment pointer JSON file
    mesh.serialize(f"{output_dir}/mesh/{object_id}.ngmesh")
    dump_json({"fragments": [f"{object_id}.ngmesh"]}, f"{output_dir}/mesh/{object_id}:0")


if __name__ == "__main__":
    main()
