"""
Command-line utility for reading connection strengths between bodies,
optionally subdivided by ROI.

Multiple CSV output formats are available:

    TODO: Give examples.
    Explain --roi and --body-fields in detail

Arguments and other options are explained below.
"""
# The above text appears in the command-line --help message.


_OUTPUT_FORMATS = ['relationships', 'inputs-outputs', 'pre-post']


def _handle_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('server')
    parser.add_argument('uuid')
    parser.add_argument('synapse_instance')
    parser.add_argument('--bodies', '-b', nargs='+', required=True)
    parser.add_argument('--rois', '-r', nargs='+', required=False)
    parser.add_argument('--body-fields', '-t', nargs='+', default=['type', 'instance'])
    parser.add_argument('--aggregate-rois', '-a', action='store_true')
    parser.add_argument('--include-nonroi', '-i', action='store_true')
    parser.add_argument('--processes', '-p', type=int, default=0)
    parser.add_argument('--output-name', '-o', required=False)
    parser.add_argument('--output-format', '-f', choices=_OUTPUT_FORMATS, default='pre-post')
    parser.add_argument('--quote-all', '-q', action='store_true', help='If given, quote all CSV values in the results.')
    parser.add_argument('--no-header', '-n', action='store_true', help='If given, omit header rows from the CSV results.')

    args = parser.parse_args()
    args.bodies = _load_bodyids(args.bodies)
    args.rois = _load_rois(args.rois)

    if not args.include_nonroi and not args.rois:
        msg = ("You must either specify at least one ROI or use --include-nonroi, "
               "otherwise the results will always be empty!")
        import sys
        print(msg, file=sys.stderr)
        sys.exit(1)

    return args


def main():
    args = _handle_args()

    # Throughout this file, we use late imports,
    # facilitating a fast --help message.
    from neuclease import configure_default_logging
    configure_default_logging()

    weights = relationship_weights(
        args.server,
        args.uuid,
        args.synapse_instance,
        args.bodies,
        args.rois,
        processes=args.processes
    )

    export_relationship_weights(
        weights,
        args.output_format,
        args.aggregate_rois,
        args.include_nonroi,
        args.output_name,
        args.quote_all,
        not args.noheader,
    )


def relationship_weights(server, uuid, synapse_instance, bodies,
                         rois=[], body_fields=['type', 'instance'], processes=0):
    """
    Download all synapse relationships (PreSynTo/PostSynTo) for the given
    bodies and determine which bodies their related synapses belong to.
    Aggregate the relationships into connection strengths (weights) for
    each unique neuron pair with unique ROI, and (if possible) add columns
    for body annotations (e.g. type, instance).

    Example results:

                 body        rel     to_body            roi  weight     type   instance  to_type       to_instance
        0       10005  PostSynTo       10013  <unspecified>       5  AOTU019  AOTU019_R   MBON01  MBON01(y5B'2a)_R
        1       10005  PostSynTo       10030  <unspecified>       6  AOTU019  AOTU019_R      NaN          (pIP1)_L
        2       10005   PreSynTo       10030  <unspecified>      16  AOTU019  AOTU019_R      NaN          (pIP1)_L
        3       10005  PostSynTo       10031  <unspecified>     211  AOTU019  AOTU019_R  AOTU041         AOTU041_R
        4       10005   PreSynTo       10031  <unspecified>       1  AOTU019  AOTU019_R  AOTU041         AOTU041_R
    """
    # Infer the names of the segmentation and body annotation instances
    seg_instance, body_annotation_instance = _find_related_instances(server, uuid, synapse_instance)

    # Point-to-point relationships
    all_rels = _fetch_all_relationship_points(
        server, uuid, synapse_instance, seg_instance, bodies, rois, processes)

    # Aggregate point -> weights
    weights = _compute_relationship_weights(all_rels)

    # Append body annotation columns if possible (e.g. type, instance, etc.)
    weights = _annotate_relationship_weights(
        server, uuid, body_annotation_instance, body_fields, weights)

    return weights


def _find_related_instances(server, uuid, synapse_instance):
    """
    Given a synapse annotation instance (dvid datatype 'annotation'),
    Determine the segmentation instance to which it is sync'd,
    and also check for a corresponding neuronjson (or keyvalue) instance
    for body annotations (containing 'type', 'instance', etc.).

    Return the names (seg_instance, body_annotation_instance)
    """
    from neuclease.dvid import fetch_repo_instances, fetch_instance_info

    repo_instances = fetch_repo_instances(server, uuid)
    instance_type = repo_instances.get(synapse_instance, 'not present')
    if instance_type != 'annotation':
        msg = f"Instance {synapse_instance} is not an annotation instance, it's {instance_type}"
        raise RuntimeError(msg)

    try:
        syn_info = fetch_instance_info(server, uuid, synapse_instance)
        seg_instance = syn_info["Base"]["Syncs"][0]
    except Exception as ex:
        msg = (f"Instance {synapse_instance} does not appear to "
               "be sync'd to a labelmap (segmentation) instance")
        raise RuntimeError(msg) from ex

    body_annotation_instance = f"{seg_instance}_annotations"
    if repo_instances.get(body_annotation_instance, None) not in ("keyvalue", "neuronjson"):
        body_annotation_instance = None

    return seg_instance, body_annotation_instance


def _fetch_all_relationship_points(server, uuid, synapse_instance, seg_instance, bodies, rois, processes):
    """
    Fetch the full set of synapse annotations for the listed bodies,
    including point relationships (partner synapses), and determine
    the bodies the partners belong to.
    If rois are provided, also determine the ROI each point pair belongs to.
    The results have one row for each 'relationship' (pre,post coordinate pair).
    """
    from functools import partial
    import pandas as pd
    from neuclease.util import compute_parallel
    from neuclease.dvid import fetch_labels_batched, determine_point_rois

    # Fetch pointwise relationships, using multiprocessing across bodies.
    _fetch_rels = partial(_fetch_body_relationships, server, uuid, synapse_instance)
    relationships = compute_parallel(_fetch_rels, bodies, processes=processes)
    relationships = [*filter(lambda r: r is not None, relationships)]
    for body, rels in zip(bodies, relationships):
        rels['body'] = body

    # Read the body associated with the 'to' side of each relationship.
    all_rels = pd.concat(relationships)
    all_rels['to_body'] = fetch_labels_batched(
        server,
        uuid,
        seg_instance,
        all_rels[['to_z', 'to_y', 'to_x']].values,
        batch_size=1000,
        processes=processes
    )
    all_rels = all_rels.reset_index()
    all_rels.columns = [*map(str.lower, all_rels.columns)]

    if len(rois) == 0:
        all_rels['roi'] = '<unspecified>'
        return all_rels

    # We need to determine the ROI for each point.
    # Here's a subtle consideration:
    # For consistency with neuprint conventions, we always use
    # the PostSyn side of a synapse pair when determining the ROI.

    # To obtain the PostSyn coordinate list, we start with a copy
    # of the 'xyz' columns and then overwrite the rows that were
    # on the PreSyn side.
    postsyn_coords = all_rels[[*'xyz']].copy()
    pre_loc = (all_rels['rel'] == "PreSynTo")
    postsyn_coords.loc[pre_loc, [*'xyz']] = all_rels.loc[pre_loc, ['to_x', 'to_y', 'to_z']].values

    # Assign points to ROIs.
    determine_point_rois(server, uuid, rois, postsyn_coords, processes=processes)
    all_rels['roi'] = postsyn_coords['roi']
    return all_rels


def _fetch_body_relationships(server, uuid, synapse_instance, body):
    """
    Fetch the table of synapse relationships belonging to a single body.
    If the body doesn't exist, then a warning is logged and None is returned.
    """
    import logging
    from requests import HTTPError
    from neuclease.dvid import fetch_annotation_label

    try:
        elements, relationships = fetch_annotation_label(
            server, uuid, synapse_instance, body, relationships=True, format='pandas')
        return relationships
    except HTTPError as ex:
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not find relationships for body {body}: {ex}")
        return None


def _compute_relationship_weights(rels):
    """
    Aggregate the given relationships table according
    to body pair, relationship (pre/post) and ROI.
    """
    groupcols = ['body', 'to_body', 'rel', 'roi']
    weights = (
        rels[groupcols]
        .value_counts()
        .rename('weight')
        .sort_index()
        .reset_index()
    )
    # Although we wanted to sort in the above order,
    # I think it's a little nicer in the output if the
    # 'rel' column appears second, between the body IDs.
    groupcols.remove('rel')
    groupcols.insert(1, 'rel')
    return weights[[*groupcols, 'weight']]


def _annotate_relationship_weights(server, uuid, body_annotation_instance, body_fields, weights):
    """
    Fetch body annotation fields (e.g. type, instance) from
    a keyvalue or neuronjson instance and append columns for
    those fields onto the given weights table.

    If body_annotation_instance is None or no body_fields were given,
    this function returns the weights table unchanged.
    """
    from neuclease.dvid import fetch_body_annotations

    body_fields = list(body_fields)
    bodies = list({*weights['body'].unique(), *weights['to_body'].unique()})

    if not body_annotation_instance or not body_fields:
        # Can't do anything
        return weights

    ann = fetch_body_annotations(server, uuid, body_annotation_instance, bodies, status_categories=None)
    for c in body_fields:
        if c not in ann:
            ann[c] = ''

    # Same as ann, but with renamed columns to match the 'to' columns
    to_ann = (ann.rename_axis('to_body')
                 .rename(columns={c: f'to_{c}' for c in ann.columns}))

    # Append body_fields as extra columns
    weights = (weights.merge(ann[body_fields], 'left', on='body')
                      .merge(to_ann[[f'to_{c}' for c in body_fields]], 'left', on='to_body'))
    return weights


def export_relationship_weights(relationship_weights, format, aggregate_rois=False, include_nonroi=True,
                                output_name=None, quote_all=False, header=True):
    """
    Export a final relationship weights table as CSV in the given format.
    """
    import os
    import csv

    assert format in _OUTPUT_FORMATS
    if output_name:
        output_name = os.path.splitext(output_name)[0]
    quoting = csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL

    if not include_nonroi:
        relationship_weights = relationship_weights.query('roi != "<unspecified>"').copy()

    if aggregate_rois:
        groupcols = [c for c in relationship_weights if c not in ['roi', 'weight']]
        relationship_weights = relationship_weights.groupby(groupcols)['weight'].sum().reset_index()

    if format == 'relationships':
        return _export_relationships(relationship_weights, output_name, header, quoting)

    if format == 'inputs-outputs':
        return _export_inputs_outputs(relationship_weights, output_name, header, quoting)

    if format == 'pre-post':
        return _export_pre_post(relationship_weights, output_name, header, quoting)


def _export_relationships(relationship_weights, output_name, header, quoting):
    cols = [c for c in relationship_weights.columns if c not in ['roi', 'weight']]
    cols = [*cols, 'roi', 'weight']
    relationship_weights = relationship_weights[cols]
    if output_name:
        relationship_weights.to_csv(f"{output_name}.csv", index=False, header=header, quoting=quoting)
    return relationship_weights


def _export_inputs_outputs(relationship_weights, output_name, header, quoting):
    from_bodyfields = [c for c in relationship_weights.columns
                       if c not in {'body', 'roi', 'rel', 'weight'} and not c.startswith('to_')]
    to_bodyfields = [c for c in relationship_weights.columns
                     if c != 'to_body' and c.startswith('to_')]

    # This is the column order FlatIron uses, assuming 'instance'
    # is the only "body field" (body annotation column).
    cols = ['body', *from_bodyfields, 'to_body', *to_bodyfields, 'weight']
    if 'roi' in relationship_weights:
        cols = ['roi', *cols]

    inputs = relationship_weights.query('rel == "PostSynTo"').reset_index(drop=True)
    outputs = relationship_weights.query('rel == "PreSynTo"').reset_index(drop=True)
    inputs = inputs[cols]
    outputs = outputs[cols]

    if output_name:
        inputs[cols].to_csv(f"{output_name}-inputs.csv", index=False, header=header, quoting=quoting)
        outputs[cols].to_csv(f"{output_name}-outputs.csv", index=False, header=header, quoting=quoting)

    return inputs, outputs


def _export_pre_post(relationship_weights, output_name, header, quoting):
    from_cols = [c for c in relationship_weights.columns
                 if c not in {'roi', 'rel', 'weight'} and not c.startswith('to_')]
    to_cols = [c for c in relationship_weights.columns
               if c.startswith('to_')]

    prepost_df = relationship_weights.copy()
    swap_loc = prepost_df.eval('rel == "PostSynTo"')
    swapped_names = dict(zip(from_cols, to_cols)) | dict(zip(to_cols, from_cols))
    prepost_df.loc[swap_loc, swapped_names.keys()] = (
        prepost_df.loc[swap_loc, swapped_names.keys()]
        .rename(columns=swapped_names)
    )
    prepost_df = (
        prepost_df
        .drop(columns=['rel'])
        .rename(columns={c: f'{c}_pre' for c in from_cols})
        .rename(columns={c: f'{c[len("to_"):]}_post' for c in to_cols})
    )

    if 'roi' in prepost_df.columns:
        prepost_df = prepost_df.drop_duplicates(['body_pre', 'body_post', 'roi'])
    else:
        prepost_df = prepost_df.drop_duplicates(['body_pre', 'body_post'])

    cols = [c for c in prepost_df.columns if c not in ['roi', 'weight']]
    cols = [*cols, 'roi', 'weight']
    prepost_df = prepost_df[cols]

    if output_name:
        prepost_df.to_csv(f"{output_name}.csv", index=False, header=header, quoting=quoting)
    return prepost_df


def _load_bodyids(body_strings):
    """
    Interpret each string in the given list as either an individual
    body ID or a path to a CSV file in which the first column is a
    whole list of body IDs.

    Return the combined list of body IDs.
    """
    from neuclease.util.csv import read_csv_col

    bodies = []
    for body_str in body_strings:
        try:
            bodies.append(int(body_str))
        except ValueError:
            bodies.extend(read_csv_col(body_str, 0))
    return bodies


def _load_rois(roi_strings):
    """
    Interpret each string in the given list as either an individual
    ROI name or a path to a CSV file in which the first column is a
    whole list of ROI names.

    Return the combined list of ROI names.
    """
    import os
    import pandas as pd

    rois = []
    for roi_str in roi_strings:
        if not os.path.isfile(roi_str):
            rois.append(roi_str)
            continue

        # Assume there's no header row when parsing the file,
        # but if the first item is 'roi', then we assume it
        # was the column header (so drop it).
        data = pd.read_csv(roi_str, header=None)
        col_data = data.iloc[:, 0].values
        if col_data[0] == 'roi':
            col_data = col_data[1:]
        rois.extend(col_data)
    return rois


if __name__ == "__main__":
    main()
