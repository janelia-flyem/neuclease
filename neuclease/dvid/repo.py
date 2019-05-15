from collections.abc import Iterable

import pandas as pd
import networkx as nx

from ..util import uuids_match
from . import dvid_api_wrapper, fetch_generic_json

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
    return fetch_generic_json(f'http://{server}/api/repos/info', session=session)

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
    r = session.post(f'http://{server}/api/repos', json=info)
    r.raise_for_status()

    repo_uuid = r.json()['root']
    return repo_uuid

@dvid_api_wrapper
def fetch_info(server, uuid, *, session=None):
    """
    Wrapper for the .../api/repo/<uuid>/info endpoint.

    See also: ``neuclease.dvid.wrapper_proxies.fetch_info()``
    """
    return fetch_generic_json(f'http://{server}/api/repo/{uuid}/info', session=session)

# Synonym
fetch_repo_info = fetch_info

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
            Optional 'tags' to initialize the instance with, e.g. "type=meshes".
            
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
    
    if tags:
        settings["Tags"] = ','.join(tags)
    
    settings.update(type_specific_settings)
    
    r = session.post(f"http://{server}/api/repo/{uuid}/instance", json=settings)
    r.raise_for_status()


@dvid_api_wrapper
def create_voxel_instance(server, uuid, instance, typename, versioned=True, compression=None, tags=[],
                          block_size=64, voxel_size=8.0, voxel_units='nanometers', background=None,
                          type_specific_settings={}, *, session=None):
    """
    Generic function ot create an instance of one of the voxel datatypes, such as uint8blk or labelmap.
    
    Note: For labelmap instances in particular, it's more convenient to call create_labelmap_instance().
    """
    assert typename in ("uint8blk", "uint16blk", "uint32blk", "uint64blk", "float32blk", "labelblk", "labelarray", "labelmap")

    if not isinstance(block_size, Iterable):
        block_size = 3*(block_size,)

    if not isinstance(voxel_size, Iterable):
        voxel_size = 3*(voxel_size,)

    block_size_str = ','.join(map(str, block_size))
    voxel_size_str = ','.join(map(str, voxel_size))

    type_specific_settings = dict(type_specific_settings)
    type_specific_settings["BlockSize"] = block_size_str
    type_specific_settings["VoxelSize"] = voxel_size_str
    type_specific_settings["VoxelUnits"] = voxel_units
    
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


def find_branch_nodes(server, repo_uuid=None, branch="", *, session=None):
    """
    Find all nodes in the repo which belong to the given branch.
    Note: By convention, the master branch is indicated by an empty branch name.
    
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

    Returns:
        list of UUIDs, sorted chronologically from first to last.
    
    
    Example:
        master_branch_uuids = find_branch_nodes('emdata3:8900', 'a77')
        current_master_uuid = master_branch_uuids[-1]
    """
    dag = fetch_repo_dag(server, repo_uuid, session=session)
    branch_uuids = nx.topological_sort(dag)
    return list(filter(lambda uuid: dag.nodes()[uuid]['Branch'] == branch, branch_uuids))

def find_master(server, repo_uuid=None):
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

    Returns:
        uuid of the most recent master branch node
    """
    master_nodes = find_branch_nodes(server, repo_uuid)
    if len(master_nodes) == 0:
        raise RuntimeError(f"Could not find master branch on server {server}, repo {repo_uuid}")
    return master_nodes[-1]


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
        If a uuids was given as a list, a pd.Series is returned.
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


def is_locked(server, uuid):
    """
    Determine whether or not the given UUID
    is locked (via fetching the repo info).
    """
    repo_info = fetch_repo_info(server, uuid)
    uuid = expand_uuid(server, uuid, repo_info=repo_info)
    return repo_info['DAG']['Nodes'][uuid]['Locked']
