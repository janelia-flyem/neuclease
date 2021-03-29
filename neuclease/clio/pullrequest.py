import logging
from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd
import networkx as nx

from ..util import find_root, tqdm_proxy as tqdm
from ..dvid import fetch_body_annotations, fetch_sizes, fetch_mutations, post_key, compute_merge_hierarchies

from . import clio_api_wrapper

logger = logging.getLogger(__name__)

@clio_api_wrapper
def fetch_pull_requests(user_email='all', *, base=None, session=None):
    """
    Fetch the pull request data submitted by a particular user,
    with light processing to make it more convenient to work with
    (e.g. ints instead of strings).

    If user_email == 'all', then pull requests for all users is returned.

    Returns:
        merge_data, timestamps
    """
    assert user_email == 'all' or '@' in user_email, \
        f"user_email does not appear to be an email address: {user_email}"

    params = {"user_email": user_email}
    r = session.get(f"{base}/v2/pull-request", params=params)
    r.raise_for_status()
    data = r.json()

    cleaned = {}
    timestamps = {}
    for email, user_data in data.items():
        cleaned[email] = {}
        timestamps[email] = {}
        for dataset, merges in user_data.items():
            # For some reason the merges are wrapped in a single list?
            assert len(merges) == 1
            merges = merges['mainToOthers']
            cleaned[email][dataset] = {}

            # Extract timestamp separately
            timestamps[email][dataset] = datetime.fromtimestamp(merges['_timestamp'])
            del merges['_timestamp']

            # Convert string keys to int keys
            cleaned_dataset = cleaned[email][dataset]
            for main, merged_bodies in merges.items():
                cleaned_dataset[int(main)] = merged_bodies

    return cleaned, timestamps


def assess_merges(dvid_server, uuid, instance, merges, mutations=None):
    """
    TODO:
    - The merges from clio are not "normalized" for us.
      A body ID can appear as both a "target" and a "fragment"
      in different entries within the same PR.
      In order to properly assess the combined result,
      it may be necessary to combine such merge sets initially.
      The tricky thing is that we would ideally prefer to return
      exactly N results for N input merges, to make it easy for the
      UI to align our results with the user's merge table.

    MORE TODO:
    - Use Ting's complete list of status priorities, rather than only hard-coding the anchor case.
    - Also check for double somas
    - Also consult a list of "forbidden merges"
      e.g. for the VNC neck break segment, or for glia, or known frankenbodies, etc.

    """
    assert all(np.issubdtype(type(k), np.integer) for k in merges.keys()), \
        ("merges should be a dict of the form:\n"
         "{main_body: [body, body, ...],\n"
         " main_body: [body, body, ...],\n"
         " ...}")

    logger.info("Fetching body annotations")
    ann_df = fetch_body_annotations(dvid_server, uuid, f'{instance}_annotations')

    logger.info("Fetching body sizes")
    all_bodies = pd.unique([*merges.keys(), *chain(*merges.values())])
    body_df = fetch_sizes(dvid_server, uuid, instance, all_bodies).to_frame()
    body_df = body_df.merge(ann_df['status'], 'left', left_index=True, right_index=True)
    body_df['status'] = body_df['status'].fillna('')

    if mutations is None:
        logger.info("Fetching mutation log")
        mutations = fetch_mutations(dvid_server, uuid, instance, dag_filter='leaf-and-parents', format='json')

    logger.info("Computing merge hierarchy forest")
    merge_forest = compute_merge_hierarchies(mutations)

    results = []
    for target, fragments in tqdm(merges.items()):
        # Fragments will each fall into one these categories
        mergeable = []         # should be applied
        already_merged = []    # should be skipped
        unknown = []           # should trigger error
        merged_elsewhere = []  # should trigger error
        anchors = []           # should trigger error
        unassessed = []        # target doesn't exist

        new_target = target
        if body_df.loc[target, 'size'] == 0:
            try:
                new_target = find_root(merge_forest, target)
            except nx.NetworkXError:
                # If this happens, something is totally wrong with the PR
                # new_target is 0, and all fragments remain unassessed
                results.append((target, 0, 'unknown', [], [], [], [], [], fragments))
                continue

        new_target_status = body_df.loc[new_target, 'status']

        frag_df = body_df.loc[fragments]

        # Handle non-existent fragments
        missing_fragments = frag_df.query("size == 0")
        for frag in missing_fragments.index:
            try:
                fragment_owner = find_root(merge_forest, frag)
            except nx.NetworkXError:
                # If this happens, something is totally wrong with the PR
                unknown.append(frag)
                continue

            if fragment_owner == new_target:
                already_merged.append(frag)
            else:
                merged_elsewhere.append(frag)

        # Handle existent fragments
        existing_fragments = frag_df.query("size > 0")
        for frag, status in existing_fragments['status'].items():
            if frag == new_target:
                already_merged.append(frag)
            elif status.lower() == "anchor":
                anchors.append(frag)
            else:
                mergeable.append(frag)

        # Each fragment should end up in exactly one category
        frag_lists = (mergeable, already_merged, unknown, merged_elsewhere, anchors, unassessed)
        listed_fragments = {*chain(*frag_lists)}
        unlisted_fragments = set(fragments) - listed_fragments
        assert not unlisted_fragments, \
            f"Some fragments were not assigned to a list: {unlisted_fragments}.\n"\
            f"Listed fragments: {frag_lists}"
        results.append((target, new_target, new_target_status, *frag_lists))

    cols = [
        'target', 'new_target', 'new_target_status',
        'mergeable', 'already_merged', 'unknown', 'merged_elsewhere', 'anchors', 'unassessed'
    ]
    results_df = pd.DataFrame(results, columns=cols)
    return results_df
