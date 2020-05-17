"""
Export the binary sparsevol masks for a group of neurons
from DVID to TIFF stack format (or PNG, etc.).


If you provide your list of neurons in a csv file with 'body' and 'group' columns,
then the neurons for each group will be combined into the same mask.

For example, this CSV file will produce two stacks:

body,group
1735382530,ORN_DA1
1672315788,ORN_DA1
1766413024,ORN_DA1
1666915286,ORN_DP1m
1697280776,ORN_DP1m
1761660556,ORN_DP1m
1666915198,ORN_DP1m

Example Usage:

    # Show usage info
    export_sparsevol --help

    # Export a mask for each group listed in neuron-groups.csv
    # If the CSV file does not have a 'group' column, each neuron is exported individually.
    export_sparsevol --scale=5 emdata4.int.janeloa.org:8900 master segmentation neuron-groups.csv

    # If you don't need grouping, body IDs can be specified directly on the command line.
    # By default, each neuron is exported individually.
    export_sparsevol -s=5 emdata4.int.janeloa.org:8900 master segmentation 1735382530 1672315788 1766413024

    # The --combine-all flag merges all neurons into one big group (only one combined mask is exported).
    export_sparsevol -s=5 --combine-all emdata4.int.janeloa.org:8900 master segmentation 1735382530 1672315788 1766413024

See below for option/parameter reference.
"""


def export_sparsevol(server, uuid, instance, neurons_df, scale=5, format='tiff', output_dir='.'):
    import os
    import vigra
    import numpy as np

    from neuclease.util import round_box, tqdm_proxy
    from neuclease.dvid import fetch_sparsevol, resolve_ref, fetch_volume_box, box_to_slicing

    uuid = resolve_ref(server, uuid)

    # Determine the segmentation bounding box at the given scale,
    # which is used as the mask shape.
    seg = (server, uuid, instance)
    box = round_box(fetch_volume_box(*seg), 64, 'out')
    box[0] = (0,0,0)
    box_scaled = box // 2**scale

    # How many digits will we need in each slice file name?
    digits = int(np.ceil(np.log10(box_scaled[1, 0])))

    # Export a mask stack for each group.
    groups = neurons_df.groupby('group', sort=False)
    num_groups = neurons_df['group'].nunique()
    group_prog = tqdm_proxy(groups, total=num_groups)
    for group, df in group_prog:
        group_prog.write(f'Group "{group}": Assembling mask')
        group_mask = np.zeros(box_scaled[1], dtype=bool)
        group_mask = vigra.taggedView(group_mask, 'zyx')

        # Overlay each body mask in the current group
        for body in tqdm_proxy(df['body'], leave=False):
            body_mask, mask_box = fetch_sparsevol(*seg, body, scale=scale, format='mask')
            group_mask[box_to_slicing(*mask_box)] |= body_mask

        # Write out the slice files
        group_prog.write(f'Group "{group}": Writing slices')
        d = f'{output_dir}/{group}.stack'
        os.makedirs(d, exist_ok=True)
        for z in tqdm_proxy(range(group_mask.shape[0]), leave=False):
            p = ('{d}/{z:' + f'0{digits}' + 'd}.{f}').format(d=d, z=z, f=format)
            vigra.impex.writeImage(group_mask[z].astype(np.uint8), p)


def main(raw_args=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scale', '-s', type=int, default=5,
                        help='Which scale to use when fetching sparsevols.')
    parser.add_argument('--combine-all', '-c', action='store_true',
                        help='Flag. If present, all bodies are treated as one large group, '
                             'regardless of the "group" column in the input (if any).')
    parser.add_argument('--format', '-f', default='tiff',
                        help='Slice file format in the output, e.g. "tiff" or "png"')
    parser.add_argument('server',
                        help='dvid server, e.g. "http://emdata4.int.janelia.org:8900"')
    parser.add_argument('uuid',
                        help='Dvid node. Branch names are also accepted (e.g. "master"), '
                             'in which case the last node in the branch is used.')
    parser.add_argument('labelmap_instance',
                        help='Name of a labelmap instance, e.g. "segmentation"')
    parser.add_argument('neuron_list', nargs='+',
                        help='List of body IDs to process. '
                             'Can be provided directly on the command line, or as a CSV file (with extension .csv). '
                             'CSV file must have a "body" column, and optionally a "group" column. '
                             'Body masks in the same group will be written to the same output stack.')
    args = parser.parse_args(raw_args)

    import pandas as pd
    from neuclease.util import csv_has_header

    if len(args.neuron_list) == 1 and args.neuron_list[0].endswith('.csv'):
        csv_path = args.neuron_list[0]
        if csv_has_header(csv_path):
            neurons_df = pd.read_csv(csv_path, header=0)
            assert 'body' in neurons_df.columns, "Your CSV file must contain a 'body' column"
            if 'group' not in neurons_df.columns:
                neurons_df['group'] = neurons_df['body']
        else:
            # We accept files without a header row, but only if they are single-column
            neurons_df = pd.read_csv(csv_path, header=None)
            assert len(neurons_df.columns) == 1, \
                "If your CSV has more than one column, you must give it a header row."

            neurons_df.columns = ['body']
            neurons_df['group'] = neurons_df['body']
    else:
        try:
            neurons = [*map(int, args.neuron_list)]
            neurons_df = pd.DataFrame({'body': neurons, 'group': neurons})
        except ValueError:
            msg = ("Did not understand neuron list.\n"
                   "If you're passing them directly on the command line, make sure they come last.\n"
                   "If you're listing them in a file, list only one file and pass it as the last argument.")
            raise RuntimeError(msg)

    if args.combine_all:
        neurons_df['group'] = 'all'

    export_sparsevol(args.server, args.uuid, args.labelmap_instance, neurons_df, args.scale, args.format)


if __name__ == "__main__":
    main()
