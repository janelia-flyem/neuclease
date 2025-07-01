import logging
from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd
import networkx as nx

from ..util import find_root, tqdm_proxy as tqdm
from ..dvid import (
    default_dvid_session, fetch_body_annotations, fetch_sizes, fetch_mutations,
    compute_merge_hierarchies, post_merge, delete_key,
    DEFAULT_BODY_STATUS_CATEGORIES)

from . import clio_api_wrapper

logger = logging.getLogger(__name__)


# These are the possible 'assessments' of merge fragments as determined in assess_merges().
# Note: This is list is ordered.
ASSESSMENT_CATEGORIES = [
    'target',
    'old_target',
    'already_absorbed_target',
    'mergeable',
    'already_merged',
    'merged_elsewhere',
    'my_target_merged_elsewhere',
    'unknown',
    'unassessed'
]


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


def assess_merges(dvid_server, uuid, instance, merges, mutations=None, include_bad_merges_in_mergeset=False):
    """
    Given a set of merges from clio, given as a dict:

        {target: [fragment, fragment, fragment, ...]}

    Produce a preliminary assessment of whether or not each merge could be applied.
    If some bodies are mentioned in multiple input entries, those entries will
    be combined and considered as a single 'mergeset', if possible.

    It's also assumed that these merges were proposed using an outdated version of the segmentation,
    so the current mutation log is fetched to determine the current "owner" of every body in the set.
    For bodies which haven't been merged into anything yet, the "owner" is simply the same as the body ID.
    But for bodies which no longer exist, the "owner" indicates which current body that object now belongs to.

    Each body will be given an "assessment" indicating the role of that body in the merges.

    The possible assessments are:

        target:
            The body is the target body in a merge set
        old_target:
            The target body was merged into something else,
            so the target's "owner" is considered the updated target.
        mergeable:
            The merge fragment still exists and can be merged into
            its target (or updated target).
        already_merged:
            The merge fragment doesn't exist any more, but it already
            belongs to the caller's intended target body (or updated target body).
        already_absorbed_target:
            The fragment still exists, but the target was already merged into it,
            i.e. in the opposite direction as the requested merge.
            Similar to 'already_merged', there is nothing to do with this fragment,
            but it may still end up being the destination (updated target) for other
            fragments in the mergeset.
        merged_elsewhere:
            The merge fragment doesn't exist any more, and it has
            already been merged to a DIFFERENT body than the caller's
            intended target body.  That's bad because it means the caller
            disagrees with the merge decisions that have been made in the head segmentation.
        my_target_merged_elsewhere:
            This is used for fragments who can't be merged because their target body
            was ALSO a fragment body (in a different row of the input),
            and the target no longer exists, and it has been merged to something different
            than the caller expected.
        unknown:
            The target body doesn't exist and its owner can't be determined.
            This indicates some major problem in the input or the dvid mutations log.
        unassessed:
            Fragments whose targets are "unknown" are not evaluated for mergeability.

    Note:
        In these results, we don't apply rules related to body status.
        To apply rules about body statuses (e.g. never merge Anchor into non-Anchor),
        see extract_and_coerce_mergeable_groups()

    TODO:
    - In the case of errors, this doesn't show the attempted target.
    - Use Ting's complete list of status priorities, rather than only hard-coding the anchor case.
    - Also check for double somas
    - Also consult a list of "forbidden merges"
      e.g. for the VNC neck break segment, or for glia, or known frankenbodies, etc.

    Args:
        include_bad_merges_in_mergeset:
            Experimental.  Assign a mergeset to every fragment, evne if its one we can't merge.
            Useful for debugging.

    Example Results:

                      size  status  exists  assessment  mergeset   owner
        body
        1                0           False     unknown        -1       0
        153127   266944916            True      target    153127  153127
        14017   1232564287  Anchor    True      target     14017   14017
        24149    360173009  Anchor    True      target     24149   24149
        11572   2314660793  Anchor    True      target     11572   11572
        10543   4777722674  Anchor    True      target     10543   10543
        12185   1915392615  Anchor    True      target     12185   12185
        69490     12112776            True      target     69490   69490
        28209    223922833            True      target     28209   28209
        39524     70728503            True      target     39524   39524
        41828     70490770            True      target     41828   41828
        16750    778461531  Anchor    True      target     16750   16750
        58207     19452724            True      target     58207   58207
        33653    124497939            True      target     33653   33653
        17500   1131998612  Anchor    True      target        -1   17500
        12254            0           False  old_target     11294   11294
        10614   5593812884  Anchor    True      target        -1   10614
        10683            0           False  old_target     10131   10131
        14021   1237311465  Anchor    True      target     14021   14021
        12269   1880149920  Anchor    True      target     12269   12269
    """
    assert all(np.issubdtype(type(k), np.integer) for k in merges.keys()), \
        ("merges should be a dict of the form:\n"
         "{main_body: [body, body, ...],\n"
         " main_body: [body, body, ...],\n"
         " ...}")

    all_bodies = pd.unique(np.array([*merges.keys(), *chain(*merges.values())]))

    logger.info("Fetching body annotations")
    ann_df = fetch_body_annotations(dvid_server, uuid, f'{instance}_annotations', all_bodies)
    for c in ['type', 'instance', 'class']:
        if c not in ann_df:
            ann_df[c] = np.nan
            ann_df[c] = np.nan
        ann_df.loc[ann_df[c] == "", c] = np.nan

    logger.info("Fetching body sizes")
    body_df = fetch_sizes(dvid_server, uuid, instance, all_bodies, processes=16).to_frame()
    body_df = body_df.merge(ann_df[['type', 'instance', 'class', 'status']], 'left', left_index=True, right_index=True)
    body_df['status'] = body_df['status'].fillna('')
    body_df['exists'] = (body_df['size'] != 0)

    if mutations is None:
        logger.info("Fetching mutation log")
        mutations = fetch_mutations(dvid_server, uuid, instance, dag_filter='leaf-and-parents', format='json')

    logger.info("Computing merge hierarchy forest")
    merge_hist = compute_merge_hierarchies(mutations)

    g = nx.DiGraph()
    for target, fragments in merges.items():
        g.add_edges_from((target, f) for f in fragments)

    def break_cycles(g):
        """
        Break cycles in a graph by removing an arbitrary edged
        from each cycle in the graph until no cycles remain.

        Works in-place.
        """
        while True:
            try:
                cycle = nx.find_cycle(g)
                g.remove_edge(*cycle[0])
            except nx.NetworkXNoCycle:
                break

    break_cycles(g)

    double_merged = [n for n, d in g.in_degree() if d > 1]
    assert not double_merged, \
        "Nonsensical input: Some fragments are listed multiple "\
        f"times, with different merge targets: {double_merged}"

    def determine_owner(b):
        """
        Use the merge history graph which was obtained from the mutations log (above),
        to determine the new "owner" of a given body, if the given body no longer exists.
        """
        if body_df.loc[b, 'exists']:
            g.nodes[b]['owner'] = b
            return

        try:
            curr = find_root(merge_hist, b)
        except nx.NetworkXError:
            # If this happens, something is totally wrong with the merge set.
            g.nodes[b]['owner'] = 0
            g.nodes[b]['assessment'] = 'unknown'
        else:
            g.nodes[b]['owner'] = curr

    for b in g.nodes():
        determine_owner(b)

    # Assess each merge by iterating over the merge targets.
    bad_merges = []
    merge_roots = [n for n, d in g.in_degree() if d == 0]
    for root in merge_roots:
        root_owner = g.nodes[root]['owner']
        if root_owner == root:
            # Overwrite the assessmemnt IFF it doesn't have one yet.
            # If it has an assessment already (as a fragment in a different merge),
            # then it can't be a target.
            g.nodes[root].setdefault('assessment', 'target')
        else:
            g.nodes[root].setdefault('assessment', 'old_target')

        # Assess each fragment in the merge
        for target, fragment in nx.dfs_edges(g, root):
            frag_owner = g.nodes[fragment]['owner']
            if root_owner == 0:
                g.nodes[fragment]['assessment'] = 'unassessed'
            elif frag_owner == root_owner == fragment:
                g.nodes[fragment]['assessment'] = 'already_absorbed_target'
            elif frag_owner == root_owner:
                g.nodes[fragment]['assessment'] = 'already_merged'
            elif frag_owner == fragment:
                g.nodes[fragment]['assessment'] = 'mergeable'
            else:
                # The fragment no longer exists, but its current owner
                # is not the same as the caller's intended owner.
                g.nodes[fragment]['assessment'] = 'merged_elsewhere'
                bad_merges.append((target, fragment))

    # Also add a unique ID and body ID as a node attributes,
    # since the caller may wish to renumber the nodes for display purposes.
    display_id = 1
    for root in merge_roots:
        for n in nx.dfs_preorder_nodes(g, root):
            g.nodes[n]['body'] = n
            g.nodes[n]['display_id'] = display_id
            display_id += 1

    if not include_bad_merges_in_mergeset:
        # Sever the bad merges
        g.remove_edges_from(bad_merges)

    merge_roots = [n for n, d in g.in_degree() if d == 0]
    for root in merge_roots:
        if g.nodes[root]['assessment'] == 'unknown':
            continue

        if g.nodes[root]['assessment'] == 'merged_elsewhere':
            if include_bad_merges_in_mergeset:
                fragments = list(nx.descendants(g, root))
            else:
                fragments = [d for d in nx.descendants(g, root) if g.nodes[d]['assessment'] == 'mergeable']
            if fragments:
                g.nodes[root]['mergeset'] = root
                for f in fragments:
                    g.nodes[f]['mergeset'] = root
                    g.nodes[f]['assessment'] = 'my_target_merged_elsewhere'

        root_owner = g.nodes[root]['owner']
        if include_bad_merges_in_mergeset:
            fragments = list(nx.descendants(g, root))
        else:
            fragments = [d for d in nx.descendants(g, root) if g.nodes[d]['assessment'] == 'mergeable']
        if fragments:
            g.nodes[root]['mergeset'] = g.nodes[root]['owner']
            for f in fragments:
                g.nodes[f]['mergeset'] = g.nodes[root]['owner']

    assessments = []
    mergesets = []
    owners = []
    for b in body_df.index:
        assessments.append(g.nodes[b]['assessment'])
        mergesets.append(g.nodes[b].get('mergeset', -1))
        owners.append(g.nodes[b]['owner'])

    body_df['assessment'] = assessments
    body_df['mergeset'] = mergesets
    body_df['owner'] = owners

    assert set(ASSESSMENT_CATEGORIES) >= set(body_df['assessment']), set(body_df['assessment'])

    body_df['assessment'] = pd.Categorical(body_df['assessment'], ASSESSMENT_CATEGORIES, ordered=True)
    return body_df, g


def extract_and_coerce_mergeable_groups(body_df, use_size=False):
    """
    Given the output from assess_merges(), above,
    extract the rows which can be used as merge sets.
    Also, apply some rules to "coerce" the merge target/fragment
    relationships into something that obeys basic body status rules
    (e.g. never merge Anchor into non-Anchor; force the merge to
    happen in the other direction.)

    All else being equal, the original "target" body is kept as the coerced target.

    The resulting DataFrame includes a new column for 'coerced_assessment',
    indicating (for each mergeset) which bodies should be used as the target
    in a merge command, and which should be the corresponding fragments.

    FIXME:
        - This example output is outdated. Nowadays this function can't return
          non-existent fragments in the results.
        - Right now, this ensures that the "best" status is kept, but this
          function should probably just reject merges between traced bodies.

    Example Output:

                           size  status  exists  assessment  mergeset        owner coerced_assessment
        body
        10131        9703514060  Anchor    True   mergeable     10131        10131             target
        10683                 0           False  old_target     10131        10131           fragment
        10543        4777722674  Anchor    True      target     10543        10543             target
        17187         775748073  Anchor    True   mergeable     10543        17187           fragment
        11294        4480310861  Anchor    True   mergeable     11294        11294             target
        12254                 0           False  old_target     11294        11294           fragment
        11572        2314660793  Anchor    True      target     11572        11572             target
        104505         76619473            True   mergeable     11572       104505           fragment
        12185        1915392615  Anchor    True      target     12185        12185             target
        42657          66182038            True   mergeable     12185        42657           fragment
        12269        1880149920  Anchor    True      target     12269        12269             target
        28561          46815471            True   mergeable     12269        28561           fragment
        14017        1232564287  Anchor    True      target     14017        14017             target
        12604        1697354396  Anchor    True   mergeable     14017        12604           fragment
        14021        1237311465  Anchor    True      target     14021        14021             target
        52509          27930713            True   mergeable     14021        52509           fragment
        16750         778461531  Anchor    True      target     16750        16750             target
        160803          7994449            True   mergeable     16750       160803           fragment
        24149         360173009  Anchor    True      target     24149        24149             target
        24034         364542554  Anchor    True   mergeable     24149        24034           fragment
        25011         327288502  Anchor    True   mergeable     24149        25011           fragment
        21815         459247723  Anchor    True   mergeable     24149        21815           fragment
        41221541903      163353            True   mergeable     24149  41221541903           fragment
        41221540810      653605            True   mergeable     24149  41221540810           fragment
        40811191220     1506353            True   mergeable     24149  40811191220           fragment
        40811387153      399685            True   mergeable     24149  40811387153           fragment
        40807848773      294972            True   mergeable     24149  40807848773           fragment
        54837          23920727            True   mergeable     24149        54837           fragment
        153168         15654750            True   mergeable     24149       153168           fragment
        45357009431       55671            True   mergeable     24149  45357009431           fragment
        45353469610       66703            True   mergeable     24149  45353469610           fragment
        41221540591       14168            True   mergeable     24149  41221540591           fragment
        41221540245       18447            True   mergeable     24149  41221540245           fragment
        40807653217      408458            True   mergeable     24149  40807653217           fragment
        12756        1634341939  Anchor    True   mergeable     28209        12756             target
        28209         223922833            True      target     28209        28209           fragment
        15801         939623164  Anchor    True   mergeable     33653        15801             target
        33653         124497939            True      target     33653        33653           fragment
        15481         813602671  Anchor    True   mergeable     39524        15481             target
        39524          70728503            True      target     39524        39524           fragment
        18551         656249050  Anchor    True   mergeable     41828        18551             target
        41828          70490770            True      target     41828        41828           fragment
        16068         621584299  Anchor    True   mergeable     58207        16068             target
        58207          19452724            True      target     58207        58207           fragment
        20446         536331788  Anchor    True   mergeable     69490        20446             target
        69490          12112776            True      target     69490        69490           fragment
        54535          24248379            True   mergeable     69490        54535           fragment
        17761         725936032  Anchor    True   mergeable    153127        17761             target
        153127        266944916            True      target    153127       153127           fragment
    """
    assert set(ASSESSMENT_CATEGORIES) >= set(body_df['assessment']), set(body_df['assessment'])
    assert set(DEFAULT_BODY_STATUS_CATEGORIES) >= set(body_df['status']), \
        "Unknown statuses: {}".format(set(body_df['status']) - set(DEFAULT_BODY_STATUS_CATEGORIES))

    body_df['assessment'] = pd.Categorical(body_df['assessment'], ASSESSMENT_CATEGORIES, ordered=True)
    body_df['status'] = pd.Categorical(body_df['status'], DEFAULT_BODY_STATUS_CATEGORIES, ordered=True)

    mergeable_assessments = ('target', 'old_target', 'mergeable')  # noqa
    mergeable_df = body_df.query('mergeset != -1 and assessment in @mergeable_assessments').copy()

    # For the purposes of determining merge direction, we only prioritize
    # between "good" statuses, not bad ones.
    assert mergeable_df['status'].dtype == "category"
    mergeable_df.loc[mergeable_df['status'] <= "", 'status'] = np.nan

    # Similarly, we consider the special class 'vnc_tbc' to be equivalent to no class at all.
    # https://flyem-cns.slack.com/archives/C02QFC68HPX/p1711620475295349?thread_ts=1710953883.337249&cid=C02QFC68HPX
    mergeable_df.loc[mergeable_df['class'] == 'vnc_tbc', 'class'] = np.nan

    # Sort by: [has_type, has_instance, ..., status, assessment]
    sortby = [
        ('mergeset', True),
    ]
    for c in [
        'type', 'flywire_type', 'hemibrain_type',
        'instance', 'group', 'manc_group', 'serial', 'mcns_serial',
        'hemilineage', 'itolee_hl', 'truman_hl', 'cell_body_fiber',
        'matching_notes', 'dimorphism',
        'class', 'superclass',
        'soma_neuromere', 'soma_side', 'fru_dsx'
    ]:
        if c in mergeable_df.columns:
            mergeable_df[f'has_{c}'] = mergeable_df[c].notnull()

            # Special case: cervical_tbd and vnc_tbc are treated as empty class.
            if c in ('class', 'superclass'):
                mergeable_df.loc[mergeable_df[c].isin(['cervical_tbd', 'vnc_tbc']), f'has_{c}'] = False

            sortby.append((f'has_{c}', False))

    sortby.append(('status', False))
    if use_size:
        sortby.append(('size', False))
    sortby.append(('body', True))

    [*by], [*ascending] = zip(*sortby)
    mergeable_df = mergeable_df.sort_values(by, ascending=ascending)
    mergeable_df = mergeable_df.drop(columns=['has_type', 'has_instance', 'has_class'], errors='ignore')

    mergeable_df['coerced_assessment'] = 'fragment'
    target_rows = mergeable_df.groupby('mergeset').head(1).index
    mergeable_df.loc[target_rows, 'coerced_assessment'] = 'target'

    # Bodies that don't exist should not be used (unless they were the target, in which case their owner may still exist).
    mergeable_df = mergeable_df.query('exists or coerced_assessment == "target"')
    mergeable_df = mergeable_df.loc[mergeable_df.groupby('mergeset').transform('size') > 1].copy()

    # Only one target per mergeset
    assert not mergeable_df.query('coerced_assessment == "target"')['mergeset'].duplicated().any()
    return mergeable_df


def apply_merges_to_owners(dvid_server, uuid, seg_instance, mergeable_df, user):
    """
    Given the output of extract_and_coerce_mergeable_groups(),
    group the bodies by the 'mergeset' column and actually apply the merges to DVID.

    The actual body IDs used in the merge command are taken from the 'owner' column.

    Also, delete the entries from the DVID's segmentation_annotations
    instance for all bodies which were merged into something else (and therefore no longer exist).

    TODO:
        Apply more status rules than just deleting the fragment metadata?
        Merge metadata into the target?
    """
    assert mergeable_df.index.name == 'body'
    assert 'owner' in mergeable_df.columns
    assert 'coerced_assessment' in mergeable_df.columns
    assert 'mergeset' in mergeable_df.columns
    assert {*mergeable_df['coerced_assessment']} == {'target', 'fragment'}

    # Make sure 'target' is the first row for each group.
    mergeable_df = mergeable_df.sort_values(['coerced_assessment'], ascending=False)

    logger.info("Fetching body annotations")
    ann_df = fetch_body_annotations(dvid_server, uuid, f"{seg_instance}_annotations", bodies=mergeable_df.index)

    s = default_dvid_session('pull-request-merge', user)

    num_mergesets = mergeable_df['mergeset'].nunique()
    logger.info(f"Sending {num_mergesets} merge commands")
    for mergeset, df in tqdm(mergeable_df.groupby('mergeset'), total=num_mergesets):
        assert df['coerced_assessment'].iloc[0] == 'target'
        assert (df['coerced_assessment'].iloc[1:] == 'fragment').all()
        assert (df.index[1:] == df['owner'].iloc[1:]).all(), \
            "Fragments that have been merged already (or merged elsewhere) are not eligible for merging!"

        target = df['owner'].iloc[0]
        fragments = df['owner'].iloc[1:]
        post_merge(dvid_server, uuid, seg_instance, target, fragments, session=s)

        # Delete corresponding annotations in the segmentation_annotations keyvalue instance
        for fragment in fragments:
            if fragment in ann_df.index:
                delete_key(dvid_server, uuid, f"{seg_instance}_annotations", str(fragment), session=s)

    logger.info("Merges applied")
