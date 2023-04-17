"""
Command-line utility for reading connection strengths between bodies,
optionally subdivided by ROI.

(Such queries are trivial in neuprint, but DVID is not structured for
connectivity queries, hence this tool.)

This tool accepts a list of bodies and returns the full set of all other
bodies with synaptic connections to the original set, both inputs
(when the original body is PostSynTo the related body) and outputs
(when the original body is PreSynTo the related body).

You may specify one or more ROIs of interest, in which case the returned
connection strengths (weights) are restricted to those which fall within
the ROIs you listed.

Using the --aggregate-rois option, the weights are summed across all listed
ROIs (and no 'roi' column is given in the results).  Otherwise, connections
within each listed ROI will be returned separately.

If the --include-nonroi option is used, then any connections outside of the
listed ROIs will also be listed, with roi stated as "<unspecified>".

In addition to obtaining pairs of connected body IDs, you can obtain extra
columns for related body fields, such as "type" or "instance",
via the --body-fields argument.

The --processes argument enables parallel fetching synapses, rois,
and point labels, but may result in worse performance for very small body lists.

--------------
OUTPUT FORMATS
--------------

The CSV table(s) can be emitted in one of three possible structures:

    --csv-format="relationships"

      Emit a table showing all bodies you listed in one column and their
      related bodies in a separate column, with a 'rel' column indicating
      the type of relationship.

      Note:

        If two of the requested bodies are synaptically related,
        then their connection will be represented *twice* in the table: once
        for the PreSynTo direction, and again for the PostSynTo direction.

      Example output columns:

         body        rel     to_body     type   instance  to_type to_instance     roi  weight
        10005   PreSynTo       10030  AOTU019  AOTU019_R      NaN    (pIP1)_L  EPA(L)      10
        10005   PreSynTo       10030  AOTU019  AOTU019_R      NaN    (pIP1)_L  LAL(L)       2
        10005   PreSynTo       10030  AOTU019  AOTU019_R      NaN    (pIP1)_L  SPS(L)       4
        10005  PostSynTo       10031  AOTU019  AOTU019_R  AOTU041   AOTU041_R  EPA(R)       1
        10005  PostSynTo       10031  AOTU019  AOTU019_R  AOTU041   AOTU041_R  LAL(R)       2

    --csv-format="inputs-outputs"

      Similar to the 'relationships' format, except the results are split into two files.
      All 'PostSynTo' rows are emitted in an 'inputs' file, and 'PreSynTo' rows are emitted
      in an 'outputs' file.  And the 'rel' column is emitted in both cases.

      Example output columns:

           roi    body     type   instance     to_body   to_type to_instance  weight
        EPA(L)   10005  AOTU019  AOTU019_R       10030       NaN    (pIP1)_L      10
        LAL(L)   10005  AOTU019  AOTU019_R       10030       NaN    (pIP1)_L       2
        SPS(L)   10005  AOTU019  AOTU019_R       10030       NaN    (pIP1)_L       4
        LAL(L)   10005  AOTU019  AOTU019_R       10041  OA-VUMa1    OA-VUMa1       6
        SPS(L)   10005  AOTU019  AOTU019_R       10041  OA-VUMa1    OA-VUMa1       2

    --csv-format="pre-post"

      Instead of listing each relationship in terms of the requested body ("body")
      and the connected body ("to_body"), swap columns as needed to emit each relationship
      in terms of "body_pre" and "body_post", regardless of which was listed by the user
      and which was found via a PreSynTo/PostSynTo relationship.

      Example output columns:

        body_pre   body_post type_pre instance_pre type_post instance_post     roi  weight
           10005       10030  AOTU019    AOTU019_R       NaN      (pIP1)_L  EPA(L)      10
           10005       10030  AOTU019    AOTU019_R       NaN      (pIP1)_L  LAL(L)       2
           10005       10030  AOTU019    AOTU019_R       NaN      (pIP1)_L  SPS(L)       4
           10031       10005  AOTU041    AOTU041_R   AOTU019     AOTU019_R  EPA(R)       1
           10031       10005  AOTU041    AOTU041_R   AOTU019     AOTU019_R  LAL(R)       2

---------------
FLATIRON FORMAT
---------------

The Flatiron team has a similar tool, whose output format can also be reproduced via this
tool if the appropriate options are specified:

    sparse_connectome \\
        emdata6.int.janelia.org:9000 \\
        94e6e \\
        synapses \\
        --bodies 10005 10031 \\
        --rois "EPA(L)" \\
        --aggregate-rois \\
        --fields instance \\
        --csv-format inputs-outputs \\
        --no-header \\
        --quote-all

    That emits two CSV files.  Example "inputs" content:

        "10005","AOTU019_R","10041","OA-VUMa1","2"
        "10005","AOTU019_R","10064","DNp57(DNp57,PS195)","3"
        "10005","AOTU019_R","10278","5-HTPMPV03_L","1"
        "10005","AOTU019_R","10546","LAL125_R","2"
        "10005","AOTU019_R","12594","WED069_L","1"
        "10005","AOTU019_R","12627","AOTU015_L","1"

---------------------
ARGUMENT DESCRIPTIONS
---------------------
"""
# The above text appears in the command-line --help message.

import logging
logger = logging.getLogger(__name__)


_OUTPUT_FORMATS = ['relationships', 'inputs-outputs', 'pre-post']


def _handle_args():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('server', help='dvid server')
    parser.add_argument('uuid', default=":master", nargs='?', help="dvid uuid")
    parser.add_argument('synapse_instance', default='synapses', nargs='?',
                        help="DVID 'annotation' instance containing synapses.")
    parser.add_argument('--bodies', '-b', nargs='+', required=True,
                        help="A list of body IDs. Can be directly listed on the command-line"
                             " or as a single-column CSV file, with or without header.")
    parser.add_argument('--rois', '-r', nargs='+', default=[],
                        help="A list of ROI compartment names. Can be directly listed on the"
                        " command-line or as a path to a single-column CSV file. (Header must be 'roi' or non-existent.)")
    parser.add_argument('--fields', '-f', nargs='+', default=['type', 'instance'],
                        help="A list of body fields to include in the results, such as 'type' or 'instance'.")
    parser.add_argument('--aggregate-rois', '-a', action='store_true',
                        help="If present, sum the weights of all listed ROIs in the results.")
    parser.add_argument('--include-nonroi', '-i', action='store_true',
                        help="If present, also include weights for synapses outside of the listed ROIs.")
    parser.add_argument('--processes', '-p', type=int, default=0,
                        help="Use multiprocessing to speed up some steps. Useful for large body lists.")
    parser.add_argument('--output-name', '-o', required=False,
                        help="Name of the output file.  File extension will be ignored and .csv will be appended.")
    parser.add_argument('--csv-format', '-c', choices=_OUTPUT_FORMATS, default='pre-post',
                        help="Structure of the output file(s).  See main help text for details.")
    parser.add_argument('--quote-all', '-q', action='store_true',
                        help='If present, quote all CSV values in the results.')
    parser.add_argument('--no-header', '-n', action='store_true',
                        help='If present, omit header rows from the CSV results.')

    args = parser.parse_args()

    args.rois = _load_rois(args.rois)

    if not args.include_nonroi and not args.rois:
        msg = ("You must either specify at least one ROI or use --include-nonroi, "
               "otherwise the results will always be empty!")
        sys.exit(msg)

    if args.rois and args.aggregate_rois and args.include_nonroi:
        msg = ("It doesn't make much sense to provide --rois if you are also "
               "using --include-nonroi and --aggregate-rois.\n"
               "You are requesting all connections and then summing their weights, "
               "so there's no need to determine ROI membership beforehand.")
        sys.exit(msg)

    args.bodies = _load_bodyids(args.bodies)
    if args.output_name:
        return args

    # Pick a default output_name
    if args.csv_format == 'inputs-outputs':
        prefix = 'connections'
    else:
        prefix = args.csv_format

    parts = [prefix]
    if len(args.bodies) == 1:
        parts.append(str(args.bodies[0]))
    if len(args.rois) == 1:
        parts.append(args.rois[0])
    args.output_name = '-'.join(parts)
    return args


def main():
    args = _handle_args()

    # Throughout this file, we use late imports,
    # facilitating a fast --help message.
    from neuclease import configure_default_logging
    configure_default_logging()

    logger.info(f"Processing {len(args.bodies)} bodies and {len(args.rois)} rois")
    weights = relationship_weights(
        args.server,
        args.uuid,
        args.synapse_instance,
        args.bodies,
        args.rois,
        args.include_nonroi,
        args.aggregate_rois,
        args.fields,
        processes=args.processes
    )

    logger.info("Writing results")
    export_relationship_weights(
        weights,
        args.csv_format,
        args.output_name,
        args.quote_all,
        not args.no_header,
    )

    logger.info("DONE")


def relationship_weights(server, uuid, synapse_instance, bodies, rois=[], include_nonroi=False,
                         aggregate_rois=False, body_fields=['type', 'instance'], processes=0):
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
    from neuclease.dvid import fetch_labels_batched

    # Infer the names of the segmentation and body annotation instances
    seg_instance, body_annotation_instance = _find_related_instances(server, uuid, synapse_instance)

    # Point-to-point relationships
    rels = _fetch_relationship_points(
        server, uuid, synapse_instance, bodies, rois, include_nonroi, processes)

    # Read the body associated with the 'to' side of each relationship.
    rels['to_body'] = fetch_labels_batched(
        server,
        uuid,
        seg_instance,
        rels[['to_z', 'to_y', 'to_x']].values,
        batch_size=1000,
        processes=processes
    )

    # Aggregate points -> weights
    rw = _compute_relationship_weights(rels, aggregate_rois)

    # Append body annotation columns if possible (e.g. type, instance, etc.)
    rw = _annotate_relationship_weights(
        server, uuid, body_annotation_instance, body_fields, rw)

    return rw


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

    logger.info(f"Using labelmap: {seg_instance}")
    logger.info(f"Using body annotations: {body_annotation_instance}")
    return seg_instance, body_annotation_instance


def _fetch_relationship_points(server, uuid, synapse_instance, bodies, rois, include_nonroi, processes):
    """
    Fetch the set of synapse annotations for the listed bodies,
    including point relationships (partner synapses).

    If rois are provided, also determine the ROI each point pair belongs to.
    If include_nonroi is False, then filter out points which fall outside of any of the listed ROIs.
    The results have one row for each 'relationship' (pre,post coordinate pair).

    The DataFrame will include columns for the body which contained
    the synapse and the ROI in which it can be found.
    """
    from functools import partial
    import pandas as pd
    from neuclease.util import compute_parallel, Timer
    from neuclease.dvid import determine_point_rois

    with Timer("Fetching synapses", logger):
        # Fetch pointwise relationships, using multiprocessing across bodies.
        _fetch_rels = partial(_fetch_body_relationships, server, uuid, synapse_instance)
        relationships = compute_parallel(_fetch_rels, bodies, processes=processes)
        relationships = [*filter(lambda r: r is not None, relationships)]
        for body, rels in zip(bodies, relationships):
            rels['body'] = body

    all_rels = pd.concat(relationships)
    all_rels = all_rels.reset_index()
    all_rels.columns = [*map(str.lower, all_rels.columns)]

    if len(rois) == 0:
        all_rels['roi'] = '<unspecified>'
    else:
        # We need to determine the ROI for each relationship.
        # Here's a subtle-yet-important consideration:
        # For consistency with neuprint conventions, we always use
        # the PostSyn side of a synapse pair when determining the ROI,
        # regardless of which side is on the "body of interest" and which is
        # on the other side of the relationship.

        # To obtain the PostSyn coordinate list, we start with a copy
        # of the 'xyz' columns and then overwrite the rows that were
        # on the PreSyn side.
        postsyn_coords = all_rels[[*'xyz']].copy()
        pre_loc = (all_rels['rel'] == "PreSynTo")
        postsyn_coords.loc[pre_loc, [*'xyz']] = all_rels.loc[pre_loc, ['to_x', 'to_y', 'to_z']].values

        # Assign points to ROIs.
        determine_point_rois(server, uuid, rois, postsyn_coords, processes=processes)
        all_rels['roi'] = postsyn_coords['roi']

    if not include_nonroi:
        all_rels = all_rels.query('roi != "<unspecified>"').reset_index(drop=True).copy()

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
        _elements, relationships = fetch_annotation_label(
            server, uuid, synapse_instance, body, relationships=True, format='pandas')
        return relationships
    except HTTPError as ex:
        # Can't use the global logger variable here
        # since this function runs in a subprocess.
        # (Loggers are not pickleable.)
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not find relationships for body {body}: {ex}")
        return None


def _compute_relationship_weights(rels, aggregate_rois):
    """
    Aggregate the given relationships table according
    to body pair, relationship (pre/post) and ROI.
    """
    groupcols = ['body', 'to_body', 'rel']
    if not aggregate_rois:
        groupcols += ['roi']

    weights = (
        rels[groupcols]
        .value_counts()
        .rename('weight')
        .sort_index()
        .reset_index()
    )

    # I think it's a little nicer in the output if the
    # 'rel' column appears second, between the body IDs.
    cols = ['body', 'rel', 'to_body'] + groupcols[3:] + ['weight']
    return weights[cols]


def _annotate_relationship_weights(server, uuid, body_annotation_instance,
                                   body_fields, relationship_weights):
    """
    Fetch body annotation fields (e.g. type, instance) from
    a keyvalue or neuronjson instance and append columns for
    those fields onto the given weights table.

    If body_annotation_instance is None or no body_fields were given,
    this function returns the weights table unchanged.
    """
    from neuclease.util import Timer
    from neuclease.dvid import fetch_body_annotations

    rw = relationship_weights
    body_fields = list(body_fields)
    bodies = list({*rw['body'].unique(), *rw['to_body'].unique()})

    if not body_annotation_instance or not body_fields:
        # Can't do anything
        return rw

    with Timer("Fetching body fields"):
        ann = fetch_body_annotations(server, uuid, body_annotation_instance,
                                     bodies, status_categories=None)
    for c in body_fields:
        if c not in ann:
            ann[c] = ''

    # Same as ann, but with renamed columns to match the 'to' columns
    to_ann = (ann.rename_axis('to_body')
                 .rename(columns={c: f'to_{c}' for c in ann.columns}))

    # Append body_fields as extra columns
    rw = (rw.merge(ann[body_fields], 'left', on='body')
            .merge(to_ann[[f'to_{c}' for c in body_fields]], 'left', on='to_body'))
    return rw


def export_relationship_weights(relationship_weights, format, output_name=None,
                                quote_all=False, header=True):
    """
    Convert a table of relationship weights to one of the above-described
    table formats, and return it as a DataFrame.
    Input must be a DataFrame in the format produced by relationship_weights().

    If an output_name is provided, export the results as CSV before returning.

    For argument meanings and example outputs, see the top-level
    help text above.
    """
    import os
    import csv

    rw = relationship_weights
    assert format in _OUTPUT_FORMATS
    quoting = csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL
    if output_name:
        output_name = os.path.splitext(output_name)[0]

    if format == 'relationships':
        return _export_relationships(rw, output_name, header, quoting)

    if format == 'inputs-outputs':
        return _export_inputs_outputs(rw, output_name, header, quoting)

    if format == 'pre-post':
        return _export_pre_post(rw, output_name, header, quoting)


def _export_relationships(relationship_weights, output_name, header, quoting):
    rw = relationship_weights
    leading_cols = [c for c in rw.columns if c not in ['roi', 'weight']]
    if 'roi' in rw.columns:
        rw = rw[[*leading_cols, 'roi', 'weight']]
    else:
        rw = rw[[*leading_cols, 'weight']]

    if output_name:
        rw.to_csv(f"{output_name}.csv", index=False, header=header, quoting=quoting)

    return rw


def _export_inputs_outputs(relationship_weights, output_name, header, quoting):
    rw = relationship_weights
    from_bodyfields = [c for c in rw.columns
                       if c not in {'body', 'roi', 'rel', 'weight'} and not c.startswith('to_')]
    to_bodyfields = [c for c in rw.columns
                     if c != 'to_body' and c.startswith('to_')]

    # This is the column order Flatiron uses, assuming 'instance'
    # is the only "body field" (body annotation column).
    cols = ['body', *from_bodyfields, 'to_body', *to_bodyfields, 'weight']
    if 'roi' in rw:
        cols = ['roi', *cols]

    inputs = rw.query('rel == "PostSynTo"').reset_index(drop=True)
    outputs = rw.query('rel == "PreSynTo"').reset_index(drop=True)
    inputs = inputs[cols]
    outputs = outputs[cols]

    if output_name:
        inputs[cols].to_csv(f"{output_name}-inputs.csv", index=False, header=header, quoting=quoting)
        outputs[cols].to_csv(f"{output_name}-outputs.csv", index=False, header=header, quoting=quoting)

    return inputs, outputs


def _export_pre_post(relationship_weights, output_name, header, quoting):
    import warnings

    from_cols = [c for c in relationship_weights.columns
                 if c not in {'roi', 'rel', 'weight'} and not c.startswith('to_')]
    to_cols = [c for c in relationship_weights.columns
               if c.startswith('to_')]

    # The results will be the same as the input, but with the
    # from/to columns swapped on all rows with a 'PostSynTo' relationship,
    # thus putting the 'pre' body first on all rows.
    prepost_df = relationship_weights.copy()
    swap_loc = prepost_df.eval('rel == "PostSynTo"')
    swapped_names = dict(zip(from_cols, to_cols)) | dict(zip(to_cols, from_cols))
    with warnings.catch_warnings():
        # https://stackoverflow.com/questions/74057367
        warnings.filterwarnings("ignore", category=FutureWarning,
                                message=".*will attempt to set the values inplace.*")
        prepost_df.loc[swap_loc, swapped_names.keys()] = (
            prepost_df.loc[swap_loc, swapped_names.keys()]
            .rename(columns=swapped_names)
        )

    # Rename columns as 'body_pre' and 'body_post', etc.
    prepost_df = (
        prepost_df
        .drop(columns=['rel'])
        .rename(columns={c: f'{c}_pre' for c in from_cols})
        .rename(columns={c: f'{c[len("to_"):]}_post' for c in to_cols})
    )

    # If two of the users' bodies turn out to be related (connected) to each other,
    # their shared connections are in the table *twice* -- so remove the duplicates.
    leading_cols = [c for c in prepost_df.columns if c not in ['roi', 'weight']]
    if 'roi' in prepost_df.columns:
        prepost_df = prepost_df.drop_duplicates(['body_pre', 'body_post', 'roi'])
        prepost_df = prepost_df[[*leading_cols, 'roi', 'weight']].reset_index(drop=True)
    else:
        prepost_df = prepost_df.drop_duplicates(['body_pre', 'body_post'])
        prepost_df = prepost_df[[*leading_cols, 'weight']].reset_index(drop=True)

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
