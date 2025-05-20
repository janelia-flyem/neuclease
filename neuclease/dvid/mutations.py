import networkx as nx

from ._dvid import dvid_api_wrapper
from .repo import resolve_ref_range, resolve_ref, fetch_repo_dag
from .kafka import kafka_msgs_to_df


@dvid_api_wrapper
def fetch_generic_mutations(server, uuid, instance, userid=None, *, action_filter=None, dag_filter='leaf-and-parents', format='pandas', session=None):
    """
    Fetch the log of successfully completed mutations.
    The log is returned in the same format as the kafka log.

    For consistency with :py:func:``read_kafka_msgs()``, this function adds
    the ``dag_filter`` and ``action_filter`` options, which are not part
    of the DVID REST API. To emulate the bare-bones /mutations results,
    use dag_filter='leaf-only'.

    As an additional convenience, a range of UUIDs can be specified in
    the ``uuid`` argument, by passing a string with special syntax as
    supported by ``resolve_ref_range()``.

    Note:
        By default, the dag_filter setting is 'leaf-and-parents'.
        So unlike the default behavior of the /mutations endpoint in the DVID REST API,
        this function returns all mutations for the given UUID and ALL of its ancestor UUIDs.
        (To achieve this, it calls the /mutations multiple times -- once per ancestor UUID.)

    Args:
        server:
            DVID server

        uuid:
            The 'leaf' UUID for which mutations should be fetched.
            Alternatively, pass a string containing (start,end) uuids
            using the syntax supported by ``resolve_ref_range()``.
            (See that function for details.)

         instance:
            A labelmap instance name.

        userid:
            If given, limit the query to only include mutations
            which were performed by the given user.
            Note: This need not be the same as the current user
            calling this function.

        action_filter:
            A list of actions to use as a filter for the returned messages.
            For example, if action_filter=['split', 'split-supervoxel'],
            all messages with other actions will be filtered out.
            (This is not part of the DVID API.  It's implemented in this
            python function a post-processing step.)

        dag_filter:
            Specifies which UUIDs for which to fetch mutations,
            relative to the specified ``uuid``.
            (This is not part of the DVID API.  It's implemented in this
            python function by calling the /mutations endpoint for multiple UUIDs.)
            This argument is ignored if ``uuid`` already specifies a reference range.

            One of:
            - 'leaf-only' (only messages whose uuid matches the one provided),
            - 'leaf-and-parents' (only messages matching the given uuid or its ancestors), or
            - None (no filtering by UUID).

        format:
            How to return the data. Either 'pandas' or 'json'.

    Returns:
        Either a DataFrame or list of parsed json values, depending
        on what you passed as 'format'.
    """
    assert dag_filter in ('leaf-only', 'leaf-and-parents', None)

    # json-values is a synonym, for compatibility with read_kafka_messages
    assert format in ('pandas', 'json', 'json-values')

    uuids = []
    if ',' in uuid:
        uuids = resolve_ref_range(server, uuid, session=session)
    elif dag_filter is None:
        dag = fetch_repo_dag(server, uuid, session=session)
        uuids = list(nx.topological_sort(dag))
    elif dag_filter == 'leaf-only':
        uuids = [resolve_ref(server, uuid, session=session)]
    elif dag_filter == 'leaf-and-parents':
        uuids = resolve_ref_range(server, f"[root, {uuid}]", session=session)

    if userid:
        params = {'userid': userid}
    else:
        params = {}

    msgs = []
    for uuid in uuids:
        r = session.get(f'{server}/api/node/{uuid}/{instance}/mutations', params=params)
        r.raise_for_status()
        msgs.extend(r.json())

    if isinstance(action_filter, str):
        action_filter = [action_filter]

    if action_filter is not None:
        action_filter = {*action_filter}
        msgs = [*filter(lambda m: m['Action'] in action_filter, msgs)]

    if format == 'pandas':
        return kafka_msgs_to_df(msgs)
    return msgs
