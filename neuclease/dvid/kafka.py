import time
import json
import getpass
import logging
from datetime import datetime

import networkx as nx

from ..util import uuids_match, Timer
from . import dvid_api_wrapper
from .server import fetch_server_info
from .repo import fetch_repo_dag
from .node import fetch_instance_info

logger = logging.getLogger(__name__)


class KafkaReadError(RuntimeError):
    pass


@dvid_api_wrapper
def read_kafka_messages(server, uuid, instance, action_filter=None, dag_filter='leaf-and-parents', return_format='json-values', group_id=None, consumer_timeout=2.0, *, session=None):
    """
    Read the stream of available Kafka messages for the given DVID instance,
    with convenient filtering options.

    This function reads from the correct kafka server(s) (as listed by DVID),
    and will (optionally) filter out messages from DVID nodes that are not
    upstream of the given UUID, or don't match the given set of dvid 'actions'.

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
            Kafka group ID to use when reading.  If not given, a new unique ID is created
            to ensure that the complete log is read.
            Note: If you re-use group IDs, then subsequent calls to this function will
                  not yield repeated messages.  The log will resume where it left off
                  from the previous call.
        
        consumer_timeout:
            Seconds to timeout (after which we assume we've read all messages).
    
    Returns:
        Filtered list of kafka ConsumerRecord or list of JSON dict (depending on return_format).
    """
    assert dag_filter in ('leaf-only', 'leaf-and-parents', None)
    assert return_format in ('records', 'json-values')

    server_info = fetch_server_info(server, session=session)
    if "Kafka Servers" not in server_info or not server_info["Kafka Servers"]:
        raise KafkaReadError(f"DVID server ({server}) does not list a kafka server")

    kafka_servers = server_info["Kafka Servers"].split(',')

    full_instance_info = fetch_instance_info(server, uuid, instance, session=session)
    data_uuid = full_instance_info["Base"]["DataUUID"]
    repo_uuid = full_instance_info["Base"]["RepoUUID"]

    logger.info(f"Reading kafka messages from {kafka_servers} for {server} / {uuid} / {instance}")
    with Timer() as timer:
        records = _read_complete_kafka_log(f'dvidrepo-{repo_uuid}-data-{data_uuid}',
                                           kafka_servers, group_id, consumer_timeout)
    logger.info(f"Reading {len(records)} kafka messages took {timer.seconds} seconds")

    # Extract and parse JSON values for easy filtering.
    values = [json.loads(rec.value) for rec in records]
    records_and_values = zip(records, values)

    # Chain filters and evaluate
    records_and_values = _filter_records_for_dag(records_and_values, dag_filter, server, repo_uuid, uuid, session)
    records_and_values = _filter_records_for_action(records_and_values, action_filter)
    records_and_values = list(records_and_values)

    if return_format == 'records':
        return [record for (record, _value) in records_and_values]
    elif return_format == 'json-values':
        return [value for (_record, value) in records_and_values]
    else:
        raise AssertionError(f"Invalid return_format: {return_format}")


def _read_complete_kafka_log(topic_name, kafka_servers, group_id=None, timeout_seconds=2.0):
    """
    Helper function.
    Read the complete kafka log for the given topic.
    Return a list of ConsumerRecord objects.
    
    Special care is taken to ensure that the complete log was read.
    An error is raised if it appears that the log was terminated early.
    """
    from pykafka import KafkaClient
    client = KafkaClient(hosts=','.join(kafka_servers))
    topic = client.topics[topic_name.encode('utf-8')]
    consumer = topic.get_simple_consumer(consumer_group=group_id, consumer_timeout_ms=int(1000*timeout_seconds))
    
    try:
        # Consumer isn't fully initialized until the first message is fetched.
        # (For example, consumer.assignment() can't be used until we fetch a message first.)
        records = [next(iter(consumer))]
    except StopIteration:
        # No messages in this topic at all.
        return []

    # Ask what the most recent message's "offset" is,
    # so we'll know for sure that we downloaded the whole log.
    end_offset = 0
    for val in topic.latest_available_offsets().values():
        end_offset = max(val.offset[0], end_offset)

    # Read all messages (until consumer timeout)
    # And read AGAIN (repeat up to MAX_TRIES) if the log appears incomplete.
    tries = 0
    MAX_TRIES = 10
    while records[-1].offset < (end_offset-1):
        records += list(consumer)
        tries += 1

        if records[-1].offset < (end_offset-1):
            if tries < MAX_TRIES:
                logger.warn(f"Could not fetch entire kafka log after {tries} tries ({records[-1].offset} / {(end_offset)})")
            else:
                # If there is an unexpected delay (e.g. a weird network/server hiccup),
                # The log may be truncated.  Raise an error in that case.
                msg = (f"Kafka log appears incomplete: \n"
                       f"Expected to log to end with offset of >= {end_offset-1}, but last "
                       f"message has offset of only {records[-1].offset}")
                raise RuntimeError(msg)

    return records


def _filter_records_for_dag(records_and_values, dag_filter, server, repo_uuid, uuid, session):
    """
    Helper function.
    Filter the given zipped records and JSON values according to the dag_filter.
    Returns an iterator.
    """
    assert dag_filter in ('leaf-only', 'leaf-and-parents', None)
    if dag_filter == 'leaf-only':
        records_and_values = filter(lambda r_v: uuids_match(r_v[1]["UUID"], uuid), records_and_values)

    elif dag_filter == 'leaf-and-parents':
        # Load DAG structure as nx.DiGraph
        dag = fetch_repo_dag(server, repo_uuid, session=session)
        
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

    return records_and_values


def _filter_records_for_action(records_and_values, action_filter):
    """
    Helper function.
    Filter the given zipped records and JSON values according to the action_filter.
    Simply select records whose actions are in the given set.
    Returns an iterator.
    """
    if action_filter is None:
        return records_and_values

    if isinstance(action_filter, str):
        action_filter = [action_filter]
    action_filter = set(action_filter)
    return filter(lambda r_v: r_v[1]["Action"] in action_filter, records_and_values)
