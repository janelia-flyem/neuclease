from collections.abc import Iterable
from functools import lru_cache

from requests import HTTPError
import pandas as pd
import networkx as nx
from asciitree import LeftAligned

from ..util import uuids_match, round_coord, find_root, tree_to_dict
from . import dvid_api_wrapper, fetch_generic_json
from .common import post_tags

VOXEL_INSTANCE_TYPENAMES = """\
float32blk
googlevoxels
labelarray
labelblk
labelmap
multichan16
rgba8blk
uint16blk
uint32blk
uint64blk
uint8blk
""".split()

INSTANCE_TYPENAMES = VOXEL_INSTANCE_TYPENAMES + """\
annotation
imagetile
keyvalue
neuronjson
labelgraph
labelsz
labelvol
roi
tarsupervoxels
""".split()

@dvid_api_wrapper
def fetch_repos_info(server, *, session=None):
    """
    Wrapper for the .../api/repos/info endpoint.
    """
    return fetch_generic_json(f'{server}/api/repos/info', session=session)

@dvid_api_wrapper
def create_repo(server, alias, description, *, session=None):
    """
    Create a new repo on the given server.
    
    Args:
        server:
            dvid server address, e.g. emdata3:8900
        
        alias:
            Short human-readable name for the repo
        
        description:
            Brief description of what you intend to use the repo for.
    
    Returns:
        Repo uuid of the newly created repo, which is also the root UUID of the repo's DAG.
    """
    info = {}
    info['alias'] = alias
    info['description'] = description
    r = session.post(f'{server}/api/repos', json=info)
    r.raise_for_status()

    repo_uuid = r.json()['root']
    return repo_uuid


@dvid_api_wrapper
def fetch_info(server, uuid=None, *, session=None):
    """
    Wrapper for the .../api/repo/<uuid>/info endpoint.

    See also: ``neuclease.dvid.wrapper_proxies.fetch_info()``
    """
    if uuid is not None:
        return fetch_generic_json(f'{server}/api/repo/{uuid}/info', session=session)
    else:
        # If there's only one repo, the user can omit the uuid.
        repos_info = fetch_repos_info(server, session=session)

        if len(repos_info) == 0:
            raise RuntimeError(f"The server {server} has no repos")
        if len(repos_info) > 1:
            raise RuntimeError(f"Cannot infer repo UUID. The server {server} has more than one repo."
                               " Please supply an explicit repo UUID.")

        # Return the first (and only) info from repos/info
        return next(iter(repos_info.values()))


# Synonym
fetch_repo_info = fetch_info


@dvid_api_wrapper
def post_info(server, uuid, info, *, session=None):
    """
    Allows changing of some repository properties by POSTing of a JSON similar to what
    you'd use in posting a new repo.  The "alias" and "description" properties can be
    optionally modified using this function.

    .. code-block:: json

        {
            "alias": "myrepo",
            "description": "This is the best repository in the universe"
        }

    Leaving out a property will keep it unchanged.

    Args:
        server:
            DVID server
        uuid:
            repo uuid
        info:
            dict, to be encoded as JSON.
    """
    assert {*info.keys()} <= {'alias', 'description'}
    assert isinstance(info.get('alias', ''), str)
    assert isinstance(info.get('description', ''), str)

    r = session.post(f'{server}/api/repo/{uuid}/info', json=info)
    r.raise_for_status()


@dvid_api_wrapper
def fetch_log(server, repo_uuid, *, session=None):
    """
    Fetch the repo log stored in DVID.

    The log is a list of strings that will be appended to the repo's log.
    They are descriptions for the entire repo and not just one node.
    For particular uuids, use node-level logging.

    Note:
        This is the repo log.  For individual node logs, see
        ``dvid.node.fetch_log()``

    Note:
        Not to be confused with other logs produced by dvid,
        such as the node note, the node log, the http log,
        the kafka log, or the mutation log.
    """
    r = session.get(f"{server}/api/repo/{repo_uuid}/log")
    r.raise_for_status()
    return r.json()["log"]


# Synonym
fetch_repo_log = fetch_log


@dvid_api_wrapper
def post_log(server, repo_uuid, messages, *, session=None):
    """
    Append messages to the repo log stored in DVID.

    The log is a list of strings that will be appended to the repo's log.
    They are descriptions for the entire repo and not just one node.
    For particular uuids, use node-level logging.

    Note:
        This is the repo log.  For individual node logs, see
        ``dvid.node.post_log()``

    Note:
        Not to be confused with other logs produced by dvid,
        such as the node note, the node log, the http log,
        the kafka log, or the mutation log.
    """
    if isinstance(messages, str):
        messages = [messages]
    assert all(isinstance(s, str) for s in messages)
    body = {"log": [*messages]}
    r = session.post(f"{server}/api/repo/{repo_uuid}/log", json=body)
    r.raise_for_status()


# Synonym
post_repo_log = post_log


@dvid_api_wrapper
def expand_uuid(server, uuid, repo_uuid=None, repo_info=None, *, session=None):
    # FIXME: Name should start with fetch_, maybe 'fetch_full_uuid()'
    """
    Given an abbreviated uuid, find the matching uuid
    on the server and return the complete uuid.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            Abbreviated uuid, e.g. `662edc`

        repo_uuid:
            The repo in which to search for the complete uuid.
            If not provided, the abbreviated uuid itself is used.
        
        repo_info:
            If you already have a copy of the repo info, you can
            pass it here to avoid an extra call to DVID.
        
    Returns:
        Complete uuid, e.g. `662edcb44e69481ea529d89904b5ef9b`
    """
    repo_uuid = repo_uuid or uuid
    if repo_info is None:
        repo_info = fetch_repo_info(server, repo_uuid, session=session)
    full_uuids = repo_info["DAG"]["Nodes"].keys()
    
    matching_uuids = list(filter(lambda full_uuid: uuids_match(uuid, full_uuid), full_uuids))
    if len(matching_uuids) == 0:
        raise RuntimeError(f"No matching uuid for '{uuid}'")
    
    if len(matching_uuids) > 1:
        raise RuntimeError(f"Multiple ({len(matching_uuids)}) uuids match '{uuid}': {matching_uuids}")

    return matching_uuids[0]


@dvid_api_wrapper
def fetch_repo_instances(server, uuid, typenames=None, *, session=None):
    """
    Fetch and parse the repo info for the given server/uuid, and
    return a dict of the data instance names and their typenames.
    Optionally filter the list to include only certain typenames.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            Abbreviated uuid, e.g. `662edc`
        
        typenames:
            Optional. A string or list-of-strings indicating which
            typenames to include in the result.  All non-matching
            data instances will be discarded from the result.

    Returns:
        dict { instance_name : typename }
    """
    if isinstance(typenames, str):
        typenames = [typenames]

    repo_info = fetch_repo_info(server, uuid, session=session)
    instance_infos = repo_info["DataInstances"].values()

    if typenames:
        typenames = set(typenames)
        instance_infos = filter(lambda info: info["Base"]["TypeName"] in typenames, instance_infos)
    
    return { info["Base"]["Name"] : info["Base"]["TypeName"] for info in instance_infos }


@dvid_api_wrapper
def create_instance(server, uuid, instance, typename, versioned=True, compression=None, tags=[], type_specific_settings={}, *, session=None):
    """
    Create a data instance of the given type.

    Note:
        Some datatypes, such as labelmap or tarsupervoxels, have their own creation functions below,
        which are more convenient than calling this function directly.

    Args:
        typename:
            Valid instance names are listed in INSTANCE_TYPENAMES

        versioned:
            Whether or not the instance should be versioned.

        compression:
            Which compression DVID should use when storing the data in the instance.
            Different instance types support different compression options.
            Typical choices are: ['none', 'snappy', 'lz4', 'gzip'].

            Note: Here, the string 'none' means "use no compression",
                  whereas a Python None value means "Let DVID choose a default compression type".

        tags:
            Optional 'tags' to initialize the instance with, either as a dict
            {name: tag} or as a list of strings ["name=tag", "name=tag", ...],
            e.g. ["type=meshes", "foo=bar"].

            Note: The DVID API allows us to provide tags in the /instance call, as query string
                  parameters, but that restricts the set of allowed strings.  Here, we post the tags
                  separately, via POST /tags, immediately after the instance is created.
                  Therefore, the given instance type must support the POST /tags endpoint.


        type_specific_settings:
            Additional datatype-specific settings to send in the JSON body.
    """
    assert typename in INSTANCE_TYPENAMES, f"Unknown typename: {typename}"

    settings = {}
    settings["dataname"] = instance
    settings["typename"] = typename

    if not versioned:
        settings["versioned"] = 'false'

    if typename == 'tarsupervoxels':
        # Will DVID return an error for us in these cases?
        # If so, we can remove these asserts...
        assert not versioned, "Instances of tarsupervoxels must be unversioned"
        assert compression in (None, 'none'), "Compression not supported for tarsupervoxels"

    if compression is not None:
        assert compression.startswith('jpeg') or compression in ('none', 'snappy', 'lz4', 'gzip')
        settings["Compression"] = compression

    settings.update(type_specific_settings)

    r = session.post(f"{server}/api/repo/{uuid}/instance", json=settings)
    r.raise_for_status()

    if tags:
        if isinstance(tags, list):
            # Convert from the old list-based format
            # to the dictionary expected by post_tags()
            tags = dict(t.split('=') for t in tags)
        post_tags(server, uuid, instance, tags, session=session)


@dvid_api_wrapper
def create_voxel_instance(server, uuid, instance, typename, versioned=True, compression=None, tags=[],
                          block_size=64, voxel_size=8.0, voxel_units='nanometers', background=None,
                          grid_store=None, min_point=None, max_point=None, type_specific_settings={},
                          *, session=None):
    """
    Generic function to create an instance of one of the voxel datatypes, such as uint8blk or labelmap.

    Args:
        grid_store:
            If provided, create this instance as a proxy for a precomputed volume hosted elsewhere.
            The name of the grid_store provided here must match one of the stores listed in the dvid
            server's TOML file.

            For example:

                [store]
                    [store.vnc-grayscale-v3]
                    engine = "ngprecomputed"
                    ref = "flyem-vnc-2-26-213dba213ef26e094c16c860ae7f4be0/v3_emdata_clahe_xy/jpeg"

    Note: For labelmap instances in particular, it's more convenient to call create_labelmap_instance().
    """
    assert typename in ("uint8blk", "uint16blk", "uint32blk", "uint64blk", "float32blk", "labelblk", "labelarray", "labelmap")

    if not isinstance(block_size, Iterable):
        block_size = 3*(block_size,)

    if not isinstance(voxel_size, Iterable):
        voxel_size = 3*(voxel_size,)

    if isinstance(voxel_units, str):
        voxel_units = 3*[voxel_units]

    block_size_str = ','.join(map(str, block_size))
    voxel_size_str = ','.join(map(str, voxel_size))
    voxel_units_str = ','.join(voxel_units)

    type_specific_settings = dict(type_specific_settings)
    type_specific_settings["BlockSize"] = block_size_str
    type_specific_settings["VoxelSize"] = voxel_size_str
    type_specific_settings["VoxelUnits"] = voxel_units_str

    if grid_store:
        type_specific_settings["GridStore"] = grid_store

    min_index = max_index = None
    if min_point is not None:
        assert len(min_point) == 3
        min_index = round_coord(min_point, block_size, 'down')
        type_specific_settings["MinPoint"] = ','.join(map(str, min_point))
        type_specific_settings["MinIndex"] = ','.join(map(str, min_index))

    if max_point is not None:
        assert len(max_point) == 3
        max_index = round_coord(max_point, block_size, 'up')
        type_specific_settings["MaxPoint"] = ','.join(map(str, max_point))
        type_specific_settings["MaxIndex"] = ','.join(map(str, max_index))

    if background is not None:
        assert typename in ("uint8blk", "uint16blk", "uint32blk", "uint64blk", "float32blk"), \
            "Background value is only valid for block-based instance types."
        type_specific_settings["Background"] = str(background)

    create_instance(server, uuid, instance, typename, versioned, compression, tags, type_specific_settings, session=session)


@dvid_api_wrapper
def fetch_repo_dag(server, uuid=None, repo_info=None, *, session=None):
    """
    Read the /repo/info for the given repo UUID and extract
    the DAG structure from it, parsed into a nx.DiGraph.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            Any node UUID within the repo of interest.
            (DVID will return the entire repo info regardless of
            which node uuid is provided here.)

        repo_info:
            If you already have a copy of the repo info, you can
            pass it here to avoid an extra call to DVID.
        
    Returns:
        The DAG as a nx.DiGraph, whose nodes' attribute
        dicts contain the fields from the DAG json data.
    
    Example:
    
        >>> g = fetch_repo_dag('emdata3:8900', 'a77')
        
        >>> list(nx.topological_sort(g))
        ['a776af0b132f44c3a428fe7607ba0da0',
         'ac90185a83a44809a2c8f0251c121827',
         '417bd9a68ed94b9381c4f35aa8a0d7f6']
        
        >>> g.nodes['ac90185a83a44809a2c8f0251c121827']
        {'Branch': '',
         'Children': [3],
         'Created': '2018-04-03T23:04:57.835725508-04:00',
         'Locked': True,
         'Log': [],
         'Note': 'Agglo Indices (updated agglo version)',
         'Parents': [1],
         'UUID': 'ac90185a83a44809a2c8f0251c121827',
         'Updated': '2018-04-04T14:15:53.377342506-04:00',
         'VersionID': 2}
    """
    if uuid is None:
        repos_info = fetch_repos_info(server, session=session)
        if len(repos_info) == 0:
            raise RuntimeError(f"The server {server} has no repos")
        if len(repos_info) > 1:
            raise RuntimeError(f"Cannot infer repo UUID. The server {server} has more than one repo."
                               " Please supply an explicit repo UUID.")

        uuid = next(iter(repos_info.keys()))
    
    if repo_info is None:
        repo_info = fetch_repo_info(server, uuid, session=session)

    # The JSON response is a little weird.
    # The DAG nodes are given as a dict with uuids as keys,
    # but to define the DAG structure, parents and children are
    # referred to by their integer 'VersionID' (not their UUID).

    # Let's start by creating an easy lookup from VersionID -> node info
    node_infos = {}
    for node_info in repo_info["DAG"]["Nodes"].values():
        version_id = node_info["VersionID"]
        node_infos[version_id] = node_info
        
    g = nx.DiGraph()
    
    # Add graph nodes (with node info from the json as the nx node attributes)
    for version_id, node_info in node_infos.items():
        g.add_node(node_info["UUID"], **node_info)
        
    # Add edges from each parent to all children
    for version_id, node_info in node_infos.items():
        parent_uuid = node_info["UUID"]
        for child_version_id in node_info["Children"]:
            child_uuid = node_infos[child_version_id]["UUID"]
            g.add_edge(parent_uuid, child_uuid)

    return g


def fetch_branches(server, repo_uuid=None, format='list', *, session=None):
    """
    Fetch the list of branches on the server. Various output formats are available.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        repo_uuid:
            Any node UUID within the repo of interest.
            (DVID will return the entire repo info regardless of
            which node uuid is provided here.)
            If the server has only one repo, this argument can be omitted.

        format:
            Either 'list', 'nx', 'pandas', 'dict', or 'text'
            Try the 'text' format, which prints a human-friendly
            diagram of the branch relationships and their UUIDs.

    Returns:
        Depends on the requested ``format``:

            - 'list': (The default) Returns a list of branch names,
              in topologically sorted order.
            - 'nx': Returns a networkx.DiGraph which has branch names
              as the node values and the dvid node info of each branch's
              FIRST NODE ONLY in the node's value.
            - 'pandas': Returns a DataFrame of the node infos,
              but includes only the FIRST NODE of each branch.
            - 'dict': Returns a nested dictionary of branch names,
              indicating the branching structure.
            - 'text': Returns a printable human-readable tree indicating
              the branching structure, showing branch names and the
              FIRST UUID of each branch.

    Example:

        .. code-block:: python

            >>> text = fetch_branches('emdata3.int.janelia.org:8900', format='dict')
            >>> print(text)
            <master> (28841)
            +-- 2020-02-18_cleave_training_setup (67ade)
            |   +-- 2020-02-18_cleave_training_setup_knechtc (9c1fd)
            |       +-- 2020-11-17_cleave-fly1 (633f6)
            |       +-- 2020-11-17_cleave-fly2 (cf6a5)
            |       +-- 2020-11-17_cleave-fly3 (afd8b)
            |       +-- 2020-12-07_cleave-cat1 (154a8)
            |       +-- 2020-12-07_cleave-cat2 (92c82)
            |       +-- 2020-12-07_cleave-cat3 (db7ea)
            |       +-- 2020-12-07_cleave-cat4 (a5e93)
            +-- 2020_orphan_link_qc_setup (86fab)
            |   +-- 2020-02-04_orphan_link_qc_shirley (b5e5c)
            +-- lou test (cd070)
            +-- mito-v1.1 (62f63)
            |   +-- test-point-neighborhoods (42197)
            |   +-- test-point-neighborhoods-2 (00f13)
            +-- test-mito-cc (da4dd)
            +-- test-mito-cc-2 (021d2)

    """
    assert format in ('list', 'nx', 'pandas', 'text', 'dict')
    dag = fetch_repo_dag(server, repo_uuid, session=session)

    branch_dag = nx.DiGraph()

    for uuid in nx.topological_sort(dag):
        branch = dag.nodes[uuid]['Branch']
        if branch not in branch_dag:
            branch_dag.add_node(branch, **dag.nodes[uuid])

    for parent, child in list(dag.edges()):
        parent_branch = dag.nodes[parent]['Branch']
        child_branch = dag.nodes[child]['Branch']
        if parent_branch != child_branch:
            branch_dag.add_edge(parent_branch, child_branch)

    if format == 'nx':
        return branch_dag

    if format == 'list':
        return sorted([*branch_dag.nodes()], key=lambda n: branch_dag.nodes[n]['VersionID'])

    if format == 'pandas':
        df = node_info_dataframe(branch_dag.nodes.values())
        return df.sort_values('VersionID')

    if format == 'dict':
        return nx.convert.to_dict_of_dicts(branch_dag)

    # TODO: Maybe show both the start and stop uuid (abc123..fed456)
    if format == 'text':
        def display(branch):
            uuid = branch_dag.nodes[branch]['UUID']
            if branch == "":
                branch = "<master>"
            return f'{branch} ({uuid[:5]})'

        d = tree_to_dict(branch_dag, find_root(branch_dag), display)
        return LeftAligned()(d)


def find_branch_nodes(server, repo_uuid=None, branch="", include_ancestors=False, *, full_info=False, session=None):
    """
    Find all nodes in the repo which belong to the given branch.

    Note:
        By convention, the master branch is indicated by an empty branch name.

    Note:
        Unlike git, each node in dvid belongs to only one branch.
        That is, when a branch is first created from a parent node,
        that parent node is not considered part of the new branch,
        nor are any of the other ancestors of the branch nodes.
        If you're interested in the all nodes that contributed to
        the history of a branch, see the ``include_ancestors`` argument.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        repo_uuid:
            Any node UUID within the repo of interest.
            (DVID will return the entire repo info regardless of
            which node uuid is provided here.)
            If the server has only one repo, this argument can be omitted.

        branch:
            Branch name to filter for.
            By default, filters for the master branch (i.e. an empty branch name).

        full_info:
            If True, return a DataFrame with columns for node attributes

        include_ancestors:
            If True, then return all nodes in the history of the branch,
            tracing all the way back to the root node.  (See note above
            regarding DVID branch conventions.)

    Returns:
        list of UUIDs, sorted chronologically from first to last,
        unless full_info=True, in which case a DataFrame is returned.


    Examples:

        >>> master_branch_uuids = find_branch_nodes('emdata3:8900', 'a77')
        >>> current_master_uuid = master_branch_uuids[-1]

        >>> master_branch_info_df = find_branch_nodes('emdata4:8900', full_info=True)
        >>> print(master_branch_info_df.columns.tolist())
        ['Branch', 'Note', 'Log', 'UUID', 'VersionID', 'Locked', 'Parents', 'Children', 'Created', 'Updated']

    """
    assert branch != "master", \
        ("Don't supply 'master' as the branch name.\n"
         "In DVID, the 'master' branch is identified via an empty string ('').")

    repo_info = fetch_repo_info(server, repo_uuid, session=session)
    dag = fetch_repo_dag(server, repo_uuid, repo_info=repo_info, session=session)
    branch_uuids = nx.topological_sort(dag)
    nodes = list(filter(lambda uuid: dag.nodes()[uuid]['Branch'] == branch, branch_uuids))

    if include_ancestors:
        leaf_node = max(nodes, key=lambda u: dag.nodes[u]['VersionID'])
        nodes = nx.ancestors(dag, leaf_node) | {leaf_node}
        nodes = sorted(nodes, key=lambda u: dag.nodes[u]['VersionID'])

    if not full_info:
        return nodes

    node_infos = [repo_info['DAG']['Nodes'][node] for node in nodes]
    return node_info_dataframe(node_infos)


def node_info_dataframe(node_infos):
    nodes_df = pd.DataFrame(node_infos)

    created = pd.to_datetime(nodes_df['Created'])
    updated = pd.to_datetime(nodes_df['Updated'])
    del nodes_df['Created']
    del nodes_df['Updated']
    nodes_df['Created'] = created
    nodes_df['Updated'] = updated

    nodes_df = nodes_df.set_index(nodes_df['UUID'].rename('uuid'))
    return nodes_df


fetch_branch_nodes = find_branch_nodes  # Alternate name


def find_branch(server, repo_uuid=None, branch="", locked_only=False):
    """
    Find the most recent node on a given branch.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        repo_uuid:
            Any node UUID within the repo of interest.
            (DVID will return the entire repo info regardless of
            which node uuid is provided here.)
            If the server has only one repo, this argument can be omitted.

        branch:
            Branch name.  By DVID convention, the master branch is indicated by the empty string.

        locked_only:
            If True, find the most recent locked UUID on the master branch.
            That is, if the leaf node is uncommitted, return its parent instead.

    Returns:
        uuid of the most recent node on the requested branch.
    """
    branch_nodes = find_branch_nodes(server, repo_uuid, branch)
    if len(branch_nodes) == 0:
        if branch == "":
            branch = "master"
        else:
            branch = f'"{branch}"'

        msg = f"Could not find {branch} branch on server {server}, repo {repo_uuid}"
        raise RuntimeError(msg)

    if locked_only and not is_locked(server, branch_nodes[-1]):
        if len(branch_nodes) == 1:
            assert branch == "", \
                "Only the master branch is capable of having fewer than 2 nodes."
            raise RuntimeError(
                "Can't find a locked master node. "
                "There's only one node, and it's uncommitted.")
        return branch_nodes[-2]
    else:
        return branch_nodes[-1]


def find_master(server, repo_uuid=None, locked_only=False):
    """
    Find the most recent master branch node.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        repo_uuid:
            Any node UUID within the repo of interest.
            (DVID will return the entire repo info regardless of
            which node uuid is provided here.)
            If the server has only one repo, this argument can be omitted.

        locked_only:
            If True, find the most recent locked UUID on the master branch.
            That is, if the leaf node is uncommitted, return its parent instead.

    Returns:
        uuid of the most recent master branch node
    """
    return find_branch(server, repo_uuid, "", locked_only)

def find_parent(server, uuids, dag=None):
    """
    Determine the parent node for one or more UUIDs.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        uuids:
            A single uuid or a list of uuids.
        dag:
            Optional. If not provided, it will be fetched from the server.

    Returns:
        If list of uuids was given, a pd.Series is returned.
        If a single uuid was given as a string, a string is returned.
    """
    if isinstance(uuids, str):
        first_uuid = uuids
    else:
        first_uuid =  uuids[0]
    
    if dag is None:
        dag = fetch_repo_dag(server, first_uuid)
    
    if isinstance(uuids, str):
        first_uuid = expand_uuid(server, first_uuid)
        return next(dag.predecessors(first_uuid))
    
    uuids = [expand_uuid(server, u) for u in uuids]
    s = pd.Series(index=uuids, data=[next(dag.predecessors(u)) for u in uuids])
    s.name = 'parent'
    s.index.name = 'child'
    return s


@lru_cache()
def find_repo_root(server, uuid=None):
    """
    Return the repo root uuid for the given dvid node.
    The result of this function is memoized.

    Args:
        server:
            dvid server
        uuid:
            dvid uuid or branch name.
    Returns:
        (str) the repo uuid
    """
    try:
        repo_info = fetch_repo_info(server, uuid)
    except Exception:
        if uuid is None:
            raise
        uuid = resolve_ref(server, uuid)
        repo_info = fetch_repo_info(server, uuid)

    return repo_info['Root']


def resolve_ref(server, ref, expand=False):
    """
    Given a ref that is either a UUID or a branch name,
    return the UUID it refers to, i.e. return the UUID
    itself OR return the last UUID of the branch.

    We also reproduce DVID's logic for resolving branch
    refs like ``:master~1`` ("master branch, 1 up one from head").

    Examples:

        >>> resolve_ref('emdata4:8900', 'abc123')
        abc123

        >>> resolve_ref('emdata4:8900', 'master')
        abc123
    """
    try:
        # Is it a uuid?
        expanded = expand_uuid(server, ref)
    except HTTPError:
        pass
    except RuntimeError as ex:
        if 'No matching uuid' in str(ex):
            pass
    else:
        if expand:
            return expanded
        return ref

    if ref.startswith(':'):
        ref = ref[1:]

    if '~' in ref:
        ref, offset = ref.split('~')
        offset = int(offset)
    else:
        offset = 0

    if ref == "master":
        ref = ""

    # Not a valid UUID.  Maybe it's a branch.
    try:
        branch_nodes = find_branch_nodes(server, branch=ref)
        if branch_nodes:
            return branch_nodes[len(branch_nodes) - offset - 1]
        raise RuntimeError(f"Could not resolve reference '{ref}'.  It is neither a UUID or a branch name.")
    except Exception as ex:
        if 'more than one repo' in ex.args[0]:
            msg = "resolve_ref() does not support servers that contain multiple repos."
            raise RuntimeError(msg) from ex
        raise


def is_locked(server, uuid):
    """
    Determine whether or not the given UUID
    is locked (via fetching the repo info).
    """
    repo_info = fetch_repo_info(server, uuid)
    uuid = expand_uuid(server, uuid, repo_info=repo_info)
    return repo_info['DAG']['Nodes'][uuid]['Locked']


def infer_lock_date(server, uuid):
    """
    Infer the date of a UUID snapshot.
    DVID records two timestamps for each UUID: 'Created' and 'Updated.
    We would like to just use 'Updated', but that timestamp changes if
    we post ANYTHING to the node, even after it was locked.
    Since it's not uncommon to write denormalizations (meshes, skeletons)
    to locked nodes, then the Updated date might not be the correct date to return.

    Here, we check the 'Created' dates of any child nodes in the DAG,
    and return the earliest date we find (including the UUID's own 'Updated' date).

    Returns:
        str
    """
    dag = fetch_repo_dag(server, uuid)
    if uuid not in dag.nodes:
        uuid = resolve_ref(server, uuid, expand=True)

    updated = dag.nodes[uuid]['Updated']
    children_created = []
    for child in dag.adj[uuid]:
        t = dag.nodes[child]['Created']
        children_created.append(t)

    print(updated)
    print(children_created)
    lock_date = min((updated, *children_created))
    return lock_date[:len('0000-00-00')]


def is_full_uuid(uuid):
    return len(uuid) == 32 and {*uuid} <= {*'0123456789abcdef'}
