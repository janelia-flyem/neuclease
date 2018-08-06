import json
import getpass
import logging
from datetime import datetime

import networkx as nx

from ..util import uuids_match, Timer
from . import dvid_api_wrapper
from .server import fetch_server_info
from .repo import fetch_and_parse_dag
from .node import fetch_instance_info

logger = logging.getLogger(__name__)

class KafkaReadError(RuntimeError):
    pass

@dvid_api_wrapper
def read_kafka_messages(server, uuid, instance, action_filter=None, dag_filter='leaf-and-parents', return_format='json-values', group_id=None, consumer_timeout=2.0, *, session=None):
    """
    Read the stream of available Kafka messages for the given DVID instance,
    and optionally filter them by UUID or Action.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid node, e.g. 'a9f2'
        
        instance:
            dvid instance name, e.g. 'segmentation'
        
        action_filter:
            A list of actions to use as a filter for the returned messages.
            For example, if action_filter=['split', 'split-complete'],
            all messages with other actions will be filtered out.

        dag_filter:
            How to filter out messages based on the UUID.
            One of:
            - 'leaf-only' (only messages whose uuid matches the one provided),
            - 'leaf-and-parents' (only messages matching the given uuid or its ancestors), or
            - None (no filtering by UUID).

        return_format:
            Either 'records' (return list of kafka ConsumerRecord objects),
            or 'json-values' (return list of parsed JSON structures from each record.value)

        group_id:
            Kafka group ID to use when reading.  If not given, a new one is created.
            (FIXME: Frequently creating new group IDs like this is probably not best-practice, but it works for now.)
        
        consumer_timeout:
            Seconds to timeout (after which we assume we've read all messages).
        
    """
    from kafka import KafkaConsumer
    
    assert dag_filter in ('leaf-only', 'leaf-and-parents', None)
    assert return_format in ('records', 'json-values')

    if group_id is None:
        # Choose a unique 'group_id' to use
        # FIXME: Frequently creating new group IDs like this is probably not best-practice, but it works for now.
        group_id = getpass.getuser() + '-' + datetime.now().isoformat()
    
    server_info = fetch_server_info(server, session=session)

    if "Kafka Servers" not in server_info or not server_info["Kafka Servers"]:
        raise KafkaReadError(f"DVID server ({server}) does not list a kafka server")

    kafka_server = server_info["Kafka Servers"]

    full_instance_info = fetch_instance_info(server, uuid, instance, session=session)
    data_uuid = full_instance_info["Base"]["DataUUID"]
    repo_uuid = full_instance_info["Base"]["RepoUUID"]

    consumer = KafkaConsumer( bootstrap_servers=[kafka_server],
                              group_id=group_id,
                              enable_auto_commit=False,
                              auto_offset_reset='earliest',
                              consumer_timeout_ms=int(consumer_timeout * 1000))

    consumer.subscribe([f'dvidrepo-{repo_uuid}-data-{data_uuid}'])

    logger.info(f"Reading kafka messages from {kafka_server} for {server} / {uuid} / {instance}")
    with Timer() as timer:
        # Read all messages (until consumer timeout)
        records = list(consumer)
    logger.info(f"Reading {len(records)} kafka messages took {timer.seconds} seconds")

    values = [json.loads(rec.value) for rec in records]
    records_and_values = zip(records, values)

    if dag_filter == 'leaf-only':
        records_and_values = filter(lambda r_v: uuids_match(r_v[1]["UUID"], uuid), records_and_values)

    elif dag_filter == 'leaf-and-parents':
        # Load DAG structure as nx.DiGraph
        dag = fetch_and_parse_dag(server, repo_uuid, session=session)
        
        # Determine full name of leaf uuid, for proper set membership
        matching_uuids = list(filter(lambda u: uuids_match(u, uuid), dag.nodes()))
        assert matching_uuids != 0, f"DAG does not contain uuid: {uuid}"
        assert len(matching_uuids) == 1, f"More than one UUID in the server DAG matches the leaf uuid: {uuid}"
        full_uuid = matching_uuids[0]
        
        # Filter based on set of leaf-and-parents
        leaf_and_parents = {full_uuid} | nx.ancestors(dag, full_uuid)
        records_and_values = filter(lambda r_v: r_v[1]["UUID"] in leaf_and_parents, records_and_values)
    else:
        assert dag_filter is None, f"Invalid choice for dag_filter: {dag_filter}"

    if action_filter is not None:
        if isinstance(action_filter, str):
            action_filter = [action_filter]
        action_filter = set(action_filter)
        records_and_values = filter(lambda r_v: r_v[1]["Action"] in action_filter, records_and_values)
    
    # Evaluate
    records_and_values = list(records_and_values)
    
    # Unzip
    if records_and_values:
        records, values = zip(*records_and_values)
    else:
        records = values = []

    if return_format == 'records':
        return records
    elif return_format == 'json-values':
        return values
    else:
        assert False
