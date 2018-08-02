import logging
from collections import namedtuple
from itertools import starmap

import numpy as np
import pandas as pd
import networkx as nx

from ...util import Timer
from .. import dvid_api_wrapper
from ..kafka import KafkaReadError, read_kafka_messages

logger = logging.getLogger(__name__)

SplitEvent = namedtuple("SplitEvent", "mutid old remain split")

@dvid_api_wrapper
def fetch_supervoxel_splits(server, uuid, instance, source='kafka', *, session=None):
    """
    Fetch supervoxel split events from dvid or kafka.
    (See fetch_supervoxel_splits_from_dvid() for details.)
    
    Note: If source='kafka', but no kafka server is found, 'dvid' is used as a fallback
    """
    assert source in ('dvid', 'kafka')

    if source == 'kafka':
        try:
            return fetch_supervoxel_splits_from_kafka(server, uuid, instance, session=session)
        except KafkaReadError:
            # Fallback to reading DVID
            source = 'dvid'

    if source == 'dvid':
        return fetch_supervoxel_splits_from_dvid(server, uuid, instance, session=session)

    assert False


@dvid_api_wrapper
def fetch_supervoxel_splits_from_dvid(server, uuid, instance, *, session=None):
    """
    Fetch the /supervoxel-splits info for the given instance.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'

    Returns:
        Dict of { uuid: event_list }, where event_list is a list of SplitEvent tuples.
        The UUIDs in the dict appear in the same order that DVID provides them in the response.
        According to the docs, they appear in REVERSE-CHRONOLOGICAL order, starting with the
        requested UUID and moving toward the repo root UUID.
        
        FIXME: It would be simpler to return a pd.DataFrame, with a column for UUID.

    WARNING:
        The /supervoxel-splits endpoint was implemented relatively recently.  It is incapable
        of returning split events for supervoxels that were split before that endpoint was
        implemented and released.  On older server, the results returned by this function may
        be incomplete, since older UUIDs predate the /supervoxel-splits endpoint support.
        However the equivalent information can be extracted from the Kafka log, albeit
        somewhat more slowly.  See fetch_supervoxel_splits_from_kafka()

    """
    # From the DVID docs:
    #
    # GET <api URL>/node/<UUID>/<data name>/supervoxel-splits
    #
    #     Returns JSON for all supervoxel splits that have occured up to this version of the
    #     labelmap instance.  The returned JSON is of format:
    # 
    #         [
    #             "abc123",
    #             [[<mutid>, <old>, <remain>, <split>],
    #             [<mutid>, <old>, <remain>, <split>],
    #             [<mutid>, <old>, <remain>, <split>]],
    #             "bcd234",
    #             [[<mutid>, <old>, <remain>, <split>],
    #             [<mutid>, <old>, <remain>, <split>],
    #             [<mutid>, <old>, <remain>, <split>]]
    #         ]
    #     
    #     The UUID examples above, "abc123" and "bcd234", would be full UUID strings and are in order
    #     of proximity to the given UUID.  So the first UUID would be the version of interest, then
    #     its parent, and so on.

    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/supervoxel-splits')
    r.raise_for_status()

    events = {}
    
    # Iterate in chunks of 2
    for uuid, event_list in zip(*2*[iter(r.json())]):
        assert isinstance(uuid, str)
        assert isinstance(event_list, list)
        events[uuid] = list(starmap(SplitEvent, event_list))
 
    return events

@dvid_api_wrapper
def fetch_supervoxel_splits_from_kafka(server, uuid, instance, actions=['split', 'split-supervoxel'], kafka_msgs=None, *, session=None):
    """
    Read the kafka log for the given instance and return a log of
    all supervoxel split events, partitioned by UUID.
    
    This produces the same output as fetch_supervoxel_splits_from_dvid(),
    but uses the kafka log instead of the DVID /supervoxel-splits endpoint.
    See the warning in that function's docstring for an explanation of why
    this function may give better results (although it is slower).

    To return all supervoxel splits, this function parses both 'split'
    and 'split-supervoxel' kafka messages.
    
    As a debugging feature, you can opt to select splits of only one
    type by specifying which actions to filter with.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'
        
        actions:
            Supervoxels can become split in two ways: body ("arbitrary") splits, and supervoxel splits.
            Normally you don't care how it was split, so you'll want the complete log,
            but as a debuggin feature you can optionally select only one type or the other via this parameter.
        
        kafka_msgs:
            The first step of this function is to fetch the kafka log, but if you've already downloaded it,
            you can provide it here.  Should be a list of parsed JSON structures.

    Returns:
        Dict of { uuid: event_list }, where event_list is a list of SplitEvent tuples.
        The UUIDs in the dict appear in CHRONOLOGICAL order (from the kafka log),
        which is the opposite ordering from fetch_supervoxel_splits_from_dvid().

    """
    assert not (set(actions) - set(['split', 'split-supervoxel'])), \
        f"Invalid actions: {actions}"
    
    if kafka_msgs is None:
        msgs = read_kafka_messages(server, uuid, instance, action_filter=actions, dag_filter='leaf-and-parents', session=session)
    else:
        msgs = list(filter(lambda msg: msg["Action"] in actions, kafka_msgs))
    
    # Supervoxels can be split via either /split or /split-supervoxel.
    # We need to parse them both.
    body_split_msgs = list(filter(lambda msg: msg['Action'] == 'split', msgs))
    sv_split_msgs = list(filter(lambda msg: msg['Action'] == 'split-supervoxel', msgs))

    events = {}
    for msg in body_split_msgs:
        if msg["SVSplits"] is None:
            logger.error(f"SVSplits is null for body {msg['Target']}")
            continue
        for old_sv_str, split_info in msg["SVSplits"].items():
            event = SplitEvent(msg["MutationID"], int(old_sv_str), split_info["Split"], split_info["Remain"])
            events.setdefault(msg["UUID"], []).append( event )

    for msg in sv_split_msgs:
        event = SplitEvent( msg["MutationID"], msg["Supervoxel"], msg["SplitSupervoxel"], msg["RemainSupervoxel"] )
        events.setdefault(msg["UUID"], []).append( event )
    
    return events


def split_events_to_graph(events):
    """
    Load the split events into an annotated networkx.DiGraph, where each node is a supervoxel ID.
    The node annotations are: 'uuid' and 'mutid', indicating the uuid and mutation id at
    the time the supervoxel was CREATED.
    
    Args:
        events: dict as returned by fetch_supervoxel_splits()
    
    Returns:
        nx.DiGraph, which will consist of trees (i.e. a forest)
    """
    g = nx.DiGraph()

    for uuid, event_list in events.items():
        for (mutid, old, remain, split) in event_list:
            g.add_edge(old, remain)
            g.add_edge(old, split)
            if 'uuid' not in g.node[old]:
                # If the old ID is not a product of a split event, we don't know when it was created.
                # (Presumably, it originates from the root UUID, but for old servers the /split-supervoxels
                # endpoint is not comprehensive all the way to the root node.)
                g.node[old]['uuid'] = '<unknown>'
                g.node[old][mutid] = -1

            g.node[remain]['uuid'] = uuid
            g.node[split]['uuid'] = uuid
            g.node[remain]['mutid'] = mutid
            g.node[split]['mutid'] = mutid
    
    return g


def find_root(g, start):
    """
    Find the root node in a tree, given as a nx.DiGraph,
    tracing up the tree starting with the given start node.
    """
    parents = [start]
    while parents:
        root = parents[0]
        parents = list(g.predecessors(parents[0]))
    return root


def extract_split_tree(events, sv_id):
    """
    Construct a nx.DiGraph from the given list of SplitEvents,
    exract the particular tree (a subgraph) that contains the given supervoxel ID.
    
    Args:
        events:
            Either a nx.DiGraph containing all split event trees of interest (so, a forest),
            or a dict of SplitEvent tuples from which a DiGraph forest can be constructed.
            
        sv_id:
            The tree containing this supervoxel ID will be extracted.
            The ID does not necessarily need to be the roof node of it's split tree.
    
    Returns:
        nx.DiGraph -- A subgraph of the forest passed in.
    """
    if isinstance(events, dict):
        g = split_events_to_graph(events)
    elif isinstance(events, nx.DiGraph):
        g = events
    else:
        raise RuntimeError(f"Unexpected input type: {type(events)}")
    
    if sv_id not in g.nodes():
        raise RuntimeError(f"Supervoxel {sv_id} is not referenced in the given split events.")
    
    root = find_root(g, sv_id)
    tree_nodes = {root} | nx.descendants(g, root)
    tree = g.subgraph(tree_nodes)
    return tree


def tree_to_dict(tree, root, display_fn=str, _d=None):
    """
    Convert the given tree (nx.DiGraph) into a dict,
    suitable for display via the asciitree module.
    
    Args:
        tree:
            nx.DiGraph
        root:
            Where to start in the tree (ancestors of this node will be ignored)
        display_fn:
            Callback used to convert node values into strings, which are used as the dict keys.
        _d:
            Internal use only.
    """
    if _d is None:
        _d = {}
    d_desc = _d[display_fn(root)] = {}
    for n in sorted(tree.successors(root)):
        tree_to_dict(tree, n, display_fn, _d=d_desc)
    return _d


def render_split_tree(tree, root=None, uuid_len=4):
    """
    Render the given split tree as ascii text.
    
    Requires the 'asciitree' module (conda install -c conda-forge asciitree)
    
    Args:
        tree:
            nx.DiGraph with 'uuid' annotations (as returned by extract_split_tree())
        root:
            Node to use as the root of the tree for display purposes.
            If None, root will be searched for.
        uuid_len:
            How many characters of the uuid to print for each node in the tree.
            Set to 0 to disable UUID display.
    
    Returns:
        str
        Example:
            >>> events = fetch_supervoxel_splits(('emdata3:8900', '52f9', 'segmentation'))
            >>> tree = extract_split_tree(events, 677527463)
            >>> print(render_split_tree(tree))
            677527463 (<unknown>)
             +-- 5813042205 (25dc)
             +-- 5813042206 (25dc)
                 +-- 5813042207 (25dc)
                 +-- 5813042208 (25dc)
    """
    root = root or find_root(tree, next(iter(tree.nodes())))

    def display_fn(n):
        uuid = tree.node[n]['uuid']
        if uuid != '<unknown>':
            uuid = uuid[:uuid_len]
        return f"{n} ({uuid})"

    if uuid_len == 0:
        display_fn = str

    d = tree_to_dict(tree, root, display_fn)

    from asciitree import LeftAligned
    return LeftAligned()(d)


def fetch_and_render_split_tree(server, uuid, instance, sv_id, split_source='kafka', *, session=None):
    """
    Fetch all split supervoxel provenance data from DVID and then
    extract the provenance tree containing the given supervoxel.
    Then render it as a string to be displayed on the console.
    """
    events = fetch_supervoxel_splits(server, uuid, instance, split_source, session=session)
    tree = extract_split_tree(events, sv_id)
    return render_split_tree(tree)


def fetch_and_render_split_trees(server, uuid, instance, sv_ids, split_source='kafka', *, session=None):
    """
    For each of the given supervoxels, produces an ascii-renderered split
    tree showing all of its ancestors,descendents, siblings, etc.

    Supervoxels that have never been involved in splits will be skipped,
    and no corresponding tree is returned.
    
    Note: If two supervoxels in the list have a common parent,
    the tree will be returned twice.
    
    TODO: It would be nice if this were smarter about eliminating
          duplicates in such cases.
    
    Returns:
        dict of { sv_id : str }
    """
    sv_ids = set(sv_ids)
    events = fetch_supervoxel_splits(server, uuid, instance, split_source, session=session)
    event_forest = split_events_to_graph(events)
    all_split_ids = set(event_forest.nodes())
    
    rendered_trees = {}
    for sv_id in sv_ids:
        if sv_id in all_split_ids:
            tree = extract_split_tree(event_forest, sv_id)
            rendered_trees[sv_id] = render_split_tree(tree)
    return rendered_trees


def fetch_split_supervoxel_sizes(server, uuid, instance, include_retired=False, split_source='kafka', *, session=None):
    """
    Fetch the list of all current split supervoxel fragments from DVID or Kafka,
    then fetch the sizes of each of those supervoxels.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'
        
        include_retired:
            If True, include 'retired' supervoxel IDs in the result.
            They will have a size of 0.
        
        split_source:
            Where to pull split events from.
            Either 'kafka' (slower) or 'dvid' (some servers return incomplete histories).
    
    Returns:
        pd.Series, indexed by SV ID
    """
    # Local import to break circular import
    from . import fetch_sizes
    
    leaf_fragment_svs, retired_svs = fetch_supervoxel_fragments(server, uuid, instance, split_source, session=session)

    with Timer(f"Fetching sizes for {len(leaf_fragment_svs)} split supervoxels", logger):
        sizes = fetch_sizes(server, uuid, instance, leaf_fragment_svs, supervoxels=True, session=session)
        sizes = np.array(sizes, np.uint32)

    sv_sizes = pd.Series(data=sizes, index=leaf_fragment_svs)
    sv_sizes.name = 'size'
    sv_sizes.index.name = 'sv'
    sv_sizes.sort_index(inplace=True)

    if include_retired:
        retired_sv_sizes = pd.Series(data=np.uint32(0), index=retired_svs)
        retired_sv_sizes.name = 'size'
        retired_sv_sizes.index.name = 'sv'
        retired_sv_sizes.sort_index(inplace=True)
        sv_sizes = pd.concat((sv_sizes, retired_sv_sizes))
        
    assert sv_sizes.index.dtype == np.uint64
    assert sv_sizes.dtype == np.uint32
    return sv_sizes


def fetch_supervoxel_fragments(server, uuid, instance, split_source='kafka', *, session=None):
    """
    Fetch the list of all supervoxels that have been split and their resulting fragments.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'
        
        split_source:
            Where to pull split events from.
            Either 'kafka' (slower) or 'dvid' (some servers return incomplete histories).
    
    Returns:
        (leaf_fragment_svs, retired_svs)
        where leaf_fragment_svs is the list of all supervoxel fragments that still exist in the instance,
        and retired_svs is the list of all supervoxels that have ever been split in the instance.
        Note that these do not constitute a mapping.
    """
    split_events = fetch_supervoxel_splits(server, uuid, instance, split_source, session=session)
    if len(split_events) == 0:
        # No splits on this node
        return (np.array([], np.uint64), np.array([], np.uint64))
    
    split_tables = list(map(lambda t: np.asarray(t, np.uint64), split_events.values()))
    split_table = np.concatenate(split_tables)

    retired_svs = split_table[:, SplitEvent._fields.index('old')]
    remain_fragment_svs = split_table[:, SplitEvent._fields.index('remain')]
    split_fragment_svs = split_table[:, SplitEvent._fields.index('split')]

    leaf_fragment_svs = (set(remain_fragment_svs) | set(split_fragment_svs)) - set(retired_svs)
    leaf_fragment_svs = np.fromiter(leaf_fragment_svs, np.uint64)
    
    return (leaf_fragment_svs, retired_svs)


def split_events_to_mapping(split_events, leaves_only=False):
    """
    Convert the given split_events,
    into a mapping, from all split fragment supervoxel IDs to their ROOT supervoxel ID,
    i.e. the supervoxel from which they came originally.

    Args:
        split_events:
            As produced by fetch_supervoxel_splits()

        leaves_only:
            If True, do not include intermediate supervoxels in the mapping;
            only include fragment IDs that have not been further split,
            i.e. they still exist in the volume.
    
    Returns:
        pd.Series, where index is fragment ID, data is root ID.
    """
    if len(split_events) == 0:
        return np.zeros((0,2), np.uint64)
    
    split_tables = list(map(lambda t: np.asarray(t, np.uint64), split_events.values()))
    split_table = np.concatenate(split_tables)

    old_svs = split_table[:, SplitEvent._fields.index('old')]
    remain_fragment_svs = split_table[:, SplitEvent._fields.index('remain')]
    split_fragment_svs = split_table[:, SplitEvent._fields.index('split')]

    if leaves_only:
        leaf_fragment_svs = (set(remain_fragment_svs) | set(split_fragment_svs)) - set(old_svs)
        fragment_svs = np.fromiter(leaf_fragment_svs, np.uint64)
    else:
        fragment_svs = np.concatenate((remain_fragment_svs, split_fragment_svs))
        
    g = split_events_to_graph(split_events)
    root_svs = np.fromiter(map(lambda sv: find_root(g, sv), fragment_svs), np.uint64, len(fragment_svs))

    mapping = pd.Series(index=fragment_svs, data=root_svs)
    mapping.index.name = 'fragment_sv'
    mapping.name = 'root_sv'
    return mapping



