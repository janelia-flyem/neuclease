import time
import logging

import ujson
import numpy as np
import pandas as pd
import networkx as nx

from ..util import uuids_match, Timer, find_root, parse_timestamp, DEFAULT_TIMESTAMP
from . import dvid_api_wrapper
from .server import fetch_server_info
from .repo import fetch_repo_dag
from .node import fetch_instance_info

logger = logging.getLogger(__name__)


class KafkaReadError(RuntimeError):
    pass


@dvid_api_wrapper
def read_kafka_messages(server, uuid, instance, action_filter=None, dag_filter='leaf-and-parents', return_format='json-values',
                        group_id=None, consumer_timeout=2.0, kafka_servers=None, topic_prefix=None, *, session=None):
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
        
        kafka_servers:
            Normally the kafka servers to read from are determined automatically
            by parsing the DVID /server/info.  But if you are attempting to read the kafka
            log of a DVID server whose kafka logging is temporarily disabled, you will need
            to specify the kafka servers here (as a list of strings).
        
        topic_prefix:
            Normally the topic prefix for all of DVID's kafka topics is determined automatically
            by parsing the DVID /server/info.  But if you know what you are doing and you want
            to force this function to use a different prefix, you can specify it here.
    
    Returns:
        Filtered list of kafka ConsumerRecord or list of JSON dict (depending on return_format).
    """
    assert dag_filter in ('leaf-only', 'leaf-and-parents', None)
    assert return_format in ('records', 'json-values')

    if kafka_servers is None or topic_prefix is None:
        server_info = fetch_server_info(server, session=session)

        if kafka_servers is None:
            if "Kafka Servers" not in server_info or not server_info["Kafka Servers"]:
                raise KafkaReadError(f"DVID server ({server}) does not list a kafka server")
            else:
                kafka_servers = server_info["Kafka Servers"].split(',')

        if topic_prefix is None:
            topic_prefix = server_info.get("Kafka Topic Prefix") or ""

    full_instance_info = fetch_instance_info(server, uuid, instance, session=session)
    data_uuid = full_instance_info["Base"]["DataUUID"]
    repo_uuid = full_instance_info["Base"]["RepoUUID"]

    # Load DAG structure as nx.DiGraph
    dag = fetch_repo_dag(server, repo_uuid, session=session)
    root_uuid = find_root(dag, repo_uuid)

    topic = f'{topic_prefix}dvidrepo-{root_uuid}-data-{data_uuid}'
    logger.info(f"Reading kafka messages for {topic} from {kafka_servers}")
    with Timer() as timer:
        records = _read_complete_kafka_log(topic, kafka_servers, group_id, consumer_timeout)
    logger.info(f"Reading {len(records)} kafka messages took {timer.seconds} seconds")

    # Extract and parse JSON values for easy filtering.
    values = [ujson.loads(rec.value) for rec in records]
    records_and_values = zip(records, values)

    # Chain filters and evaluate
    records_and_values = _filter_records_for_dag(records_and_values, dag_filter, dag, uuid)
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
    consumer = topic.get_simple_consumer( consumer_group=group_id,
                                          consumer_timeout_ms=int(1000*timeout_seconds),
                                          auto_commit_enable=(group_id is not None) )
    
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

    # Avoid ReferenceError in pykafka.
    # See comment in https://github.com/Parsely/pykafka/pull/827
    consumer.stop()
    time.sleep(0.1)

    return records


def _filter_records_for_dag(records_and_values, dag_filter, dag, uuid):
    """
    Helper function.
    Filter the given zipped records and JSON values according to the dag_filter.
    Returns an iterator.
    """
    assert dag_filter in ('leaf-only', 'leaf-and-parents', None)
    if dag_filter == 'leaf-only':
        records_and_values = filter(lambda r_v: uuids_match(r_v[1]["UUID"], uuid), records_and_values)

    elif dag_filter == 'leaf-and-parents':
        
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


def kafka_msgs_to_df(msgs, drop_duplicates=False, default_timestamp=DEFAULT_TIMESTAMP):
    """
    Load the messages into a DataFrame with columns for
    timestamp, uuid, mut_id (if present), key (if present), and msg (the complete message).
    
    Note: See `neuclease.util.DEFAULT_TIMESTAMP`.
    (At the time of this writing, `2018-01-01`).
    
    Args:
        msgs:
            JSON messages, either as strings or pre-parsed into a list-of-dicts.
            
        drop_duplicates:
            If the kafka messages contain exact duplicates for some reason,
            this option can be used to drop the duplicates from the result.
            Use with caution, especially if you are trying to disambiguate
            messages from a 'mirror' server.  (Especially 'complete' messages,
            which might be *correctly* duplicated.)
        
        default_timestamp:
            Any messages that lack a 'Timestamp' field will have
            their 'timestamp' column set to this in the output dataframe.

    Returns:
        DataFrame
    """
    if len(msgs) == 0:
        return pd.DataFrame([], columns=['timestamp', 'uuid', 'mutid', 'msg'])
        
    if drop_duplicates:
        if isinstance(msgs[0], str):
            msg_strings = msgs
        else:
            # Convert to strings for hashing
            msg_strings = [ujson.dumps(msg, sort_keys=True) for msg in msgs]
        
        msg_strings = pd.Series(msg_strings)
        msg_strings.drop_duplicates(inplace=True)
        msgs = msg_strings.values

    # Parse JSON if necessary    
    if isinstance(msgs[0], str):
        msgs = [ujson.loads(msg) for msg in msgs]
    
    timestamps = np.repeat(None, len(msgs))
    timestamps[:] = default_timestamp

    for i, msg in enumerate(msgs):
        if 'Timestamp' in msg:
            timestamps[i] = msg['Timestamp'][:len('2018-01-01 00:00:00.000')]

    msgs_df = pd.DataFrame({'msg': msgs})
    msgs_df['timestamp'] = timestamps
    msgs_df['timestamp'] = pd.to_datetime(msgs_df['timestamp'])
    msgs_df['uuid'] = [msg['UUID'] for msg in msgs_df['msg']]
    msgs_df['uuid'] = msgs_df['uuid'].astype('category')
    
    if 'MutationID' in msgs[0]:
        mutids = []
        for msg in msgs_df['msg']:
            try:
                mutids.append( msg['MutationID'] )
            except KeyError:
                mutids.append( 0 )
        msgs_df['mutid'] = mutids
    
    if 'Key' in msgs[0]:
        msgs_df['key'] = [msg['Key'] for msg in msgs_df['msg']]

    columns = ['timestamp', 'uuid', 'mutid', 'key', 'msg']

    if 'mutid' not in msgs_df.columns:
        columns.remove('mutid')

    if 'key' not in msgs_df.columns:
        columns.remove('key')
    
    return msgs_df[columns]


def filter_kafka_msgs_by_timerange(kafka_msgs, min_timestamp=None, max_timestamp=None, min_mutid=None, max_mutid=None):
    """
    Given a list (or dataframe) of kafka messages from a DVID data instance,
    filter messages according to a mutation ID range and/or a timestamp range.

    For example, this call removes messages except those with at least
    mutation ID 1002213701 and occurred no later than "2018-10-16 13:00":
    
        all_msgs = read_kafka_messages('emdata3:8900', 'ef1d', 'segmentation')
        filtered_msgs = filter_kafka_msgs_by_timerange(all_msgs, min_mutid=1002213701, max_timestamp='2018-10-16 13:00')
    
    Args:
        kafka_msgs:
            Either a list of JSON messages from a dvid instance kafka log,
            or a DataFrame of kafka messages as returned by kafka_msgs_to_df()
        
        min_timestamp, max_timestamp:
            min/max timestamps, as a datetime or string
        
        min_mutid, max_mutid:
            min/max DVID mutation IDs (integers)
    
    Returns:
        Filtered list or DataFrame (depending on input type)
    """
    queries = []
    if min_timestamp is not None:
        min_timestamp = parse_timestamp(min_timestamp)
        queries.append('timestamp >= @min_timestamp')
    
    if max_timestamp is not None:
        max_timestamp = parse_timestamp(max_timestamp)
        queries.append('timestamp <= @max_timestamp')
    
    if min_mutid is not None:
        queries.append('mutid >= @min_mutid')
        
    if max_mutid is not None:
        queries.append('mutid <= @max_mutid')

    if not queries:
        return kafka_msgs

    if isinstance(kafka_msgs, pd.DataFrame):
        kafka_df = kafka_msgs
    else:
        kafka_df = kafka_msgs_to_df(kafka_msgs)

    q = ' and '.join(queries)
    kafka_df = kafka_df.query(q)

    if isinstance(kafka_msgs, pd.DataFrame):
        return kafka_df
    else:    
        return kafka_df['msg'].tolist()


