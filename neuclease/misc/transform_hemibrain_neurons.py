import os
import sys
import json
import argparse
import tempfile
import subprocess
from functools import partial

Hemibrain_v12 = ('emdata4:8900', '31597d95bd844060b0ccc928a1a8a0a4')

# John Bogovic's programs to transform hemibrain points into unisex template and unisex to CNS
#Hemibrain_to_unisex = '/nrs/saalfeld/john/flyem_maleBrain/transformation_2022Apr07/csv_hemibrainNm-JRC2018Uum'
#Unisex_to_cns = '/nrs/saalfeld/john/flyem_maleBrain/transformation_2022Apr07/csv_JRC2018Uum-to-EMnm'

Hemibrain_to_unisex = '/groups/flyem/data/scratchspace/flyemflows/cns-brain/alignment/csv_hemibrainNm-JRC2018Uum'
Unisex_to_cns = '/nrs/saalfeld/john/flyem_maleBrain/transformation_2022July11/csv_JRC2018Uum-to-EMnm'


UNISEX_MIDLINE_MICRONS = 313.5

SWC_COLUMNS = ['node', 'kind', 'x', 'y', 'z', 'radius', 'parent']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-space', '-t', choices=['unisex-template', 'male-cns'], default='male-cns')
    parser.add_argument('--processes', '-p', type=int, default=2)
    parser.add_argument('--format', '-f', choices=['swc-voxels', 'neuroglancer'], default='neuroglancer')
    parser.add_argument('--reflect', action='store_true', default=False, help='reflect the skeletons in template space')
    parser.add_argument('--skip-existing', action='store_true', default=False, help='Skip neurons which already have results in the output directory')
    parser.add_argument('body_csv')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    from neuclease import configure_default_logging
    configure_default_logging()

    transform_hemibrain_neurons(args)
    print("DONE")


def transform_hemibrain_neurons(args):
    from neuclease.util import read_csv_col, tqdm_proxy, compute_parallel
    from neuclease.dvid import fetch_skeleton

    bodies = read_csv_col(args.body_csv)
    tmpdir = tempfile.mkdtemp()
    print(f"Working in {tmpdir}")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.format == 'neuroglancer':
        with open(f'{args.output_dir}/info', 'w') as f:
            info = {"@type": "neuroglancer_skeletons"}
            json.dump(info, f, indent=2)

    fn = partial(_transform_hemibrain_neuron, args, tmpdir)
    compute_parallel(fn, bodies, processes=args.processes)


def _transform_hemibrain_neuron(args, tmpdir, body):
    from neuclease.dvid import fetch_skeleton
    from neuclease.util import skeleton_to_neuroglancer
    from requests import HTTPError

    output_path = f'{args.output_dir}/{body}'
    if args.format == 'swc-voxels':
        output_path += '.swc'
    
    if args.skip_existing and os.path.exists(output_path):
        return

    try:
        hemi_df = fetch_skeleton(*Hemibrain_v12, 'segmentation_skeletons', body, 'pandas')
    except HTTPError as ex:
        print(f"Failed to fetch skeleton for body {body}", file=sys.stderr)
        return None

    unisex_df = transform_hemibrain_neuron_to_unisex(hemi_df, None, tmpdir, body)

    if args.reflect:
        midline = UNISEX_MIDLINE_MICRONS * 1000 / 8
        unisex_df['x'] = (unisex_df.eval('2 * @midline - x'))

    assert args.target_space in ('male-cns', 'unisex-template')
    if args.target_space == 'male-cns':
        output_df = transform_unisex_neuron_to_cns(unisex_df, None, tmpdir, body)
    elif target_space == 'unisex-template':
        output_df = unisex_df

    if args.format == 'neuroglancer':
        skeleton_to_neuroglancer(output_df, 8, output_path)
    else:
        output_df.to_csv(output_path, sep=' ', index=False, header=False)


def transform_hemibrain_neuron_to_unisex(hemi_df, output_path=None, tmpdir=None, body=0):
    """
    Transform a hemibrain neuron to unisex template space.
    The resulting skeleton is in voxel units, per FlyEM/DVID conventions.
    """
    import pandas as pd

    if not tmpdir:
        tmpdir = tempfile.mkdtemp()
    hemi_csv = f'{tmpdir}/{body}-hemi.csv'
    unisex_csv = f'{tmpdir}/{body}-unisex.csv'

    ## Convert to nm [edit: nevermind, this program expects voxel units]
    #hemi_df = hemi_df.copy(deep=True)
    #hemi_df[[*'xyz']] *= 8
    hemi_df[[*'xyz']].to_csv(hemi_csv, header=False, index=False)

    # Run John's conversion
    # Note: result is in microns, not nm or voxels.
    try:
        subprocess.run([Hemibrain_to_unisex, hemi_csv, unisex_csv], check=True, capture_output=True)
    except subprocess.CalledProcessError as ex:
        print(ex.stdout.decode('utf-8'), '\n', ex.stderr.decode('utf-8'), file=sys.stderr)
        raise

    unisex_df = pd.read_csv(unisex_csv, header=None, names=[*'xyz'])

    # Copy other columns and convert back to voxels
    unisex_df[['node', 'kind', 'radius', 'parent']] = hemi_df[['node', 'kind', 'radius', 'parent']]
    unisex_df[[*'xyz']] *= 1000/8
    unisex_df = unisex_df[SWC_COLUMNS]

    if output_path:
        unisex_df.to_csv(output_path, sep=' ', index=False, header=False)

    return unisex_df


def transform_unisex_neuron_to_cns(unisex_df, output_path=None, tmpdir=None, body=0):
    """
    Transform a unisex neuron to CNS template space.
    We assume the input is in voxel units, per FlyEM/DVID conventions,
    and the result will also be in voxel units.
    """
    import pandas as pd

    if not tmpdir:
        tmpdir = tempfile.mkdtemp()
    unisex_csv = f'{tmpdir}/{body}-unisex.csv'
    cns_csv = f'{tmpdir}/{body}-cns.csv'

    # Convert to microns
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
    
    cns_df = pd.read_csv(cns_csv, header=None, names=[*'xyz'])

    # Copy other columns and convert back to voxels
    cns_df[['node', 'kind', 'radius', 'parent']] = unisex_df[['node', 'kind', 'radius', 'parent']]
    cns_df[[*'xyz']] /= 8
    cns_df = cns_df[SWC_COLUMNS]

    if output_path:
        cns_df.to_csv(output_path, sep=' ', index=False, header=False)

    return cns_df
    

if __name__ == "__main__":
    main()
