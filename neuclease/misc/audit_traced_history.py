import logging

import numpy as np
import pandas as pd

from neuclease.util import tqdm_proxy
from neuclease.dvid import (read_kafka_messages, read_labelmap_kafka_df, kafka_msgs_to_df,
                            fetch_body_annotations, find_branch_nodes, expand_uuid,
                            fetch_size, fetch_sizes, fetch_supervoxel_splits, fetch_retired_supervoxel_size)

logger = logging.getLogger(__name__)

TRACED_STATUSES = {'Leaves', 'Prelim Roughly traced', 'Roughly traced', 'Traced'}

def audit_traced_history_for_branch(server, branch='', instance='segmentation', start_uuid=None, final_uuid=None):
    """
    Audits a labelmap kafka log and dvid history to
    check for the following spurious events:
    
        Merges:
            - A traced body should never be merged into another
              non-traced body and lose its body id
            - Two traced bodies that are merged together should be flagged — this
              might happen if a Leaves is merged to a Roughly traced
        
        Cleaves/body-splits:
            - A traced body should never be split where the bigger
              piece (or close to half the size) is split off

    Note:
        This function does not yet handle split events,
        which leads to certain cases that will be missed here.
        Furthermore, the body sizes in the results are not guaranteed to be accurate.
    """
    if start_uuid is not None:
        start_uuid = expand_uuid(server, start_uuid)

    if final_uuid is not None:
        final_uuid = expand_uuid(server, final_uuid)

    branch_uuids = find_branch_nodes(server, final_uuid, branch)
    leaf_uuid = branch_uuids[-1]
    
    if start_uuid is None:
        # Start with the second uuid by default
        start_uuid = branch_uuids[1]
    else:
        assert start_uuid != branch_uuids[0], \
            "Can't start from the root uuid, since the size/status information before that is unknown."

    if final_uuid is None:
        final_uuid = branch_uuids[-1]

    start_uuid_index = branch_uuids.index(start_uuid)
    final_uuid_index = branch_uuids.index(final_uuid)
    prev_uuid = branch_uuids[start_uuid_index-1]
    
    audit_uuids = branch_uuids[start_uuid_index:1+final_uuid_index]
    msgs_df = read_labelmap_kafka_df(server, final_uuid, instance, drop_completes=True)
    
    ann_kv_log = read_kafka_messages(server, final_uuid, f'{instance}_annotations')
    ann_kv_log_df = kafka_msgs_to_df(ann_kv_log)
    ann_kv_log_df['body'] = ann_kv_log_df['key'].astype(np.uint64)

    ann_df = fetch_body_annotations(server, prev_uuid, f'{instance}_annotations')[['status']]

    split_events = fetch_supervoxel_splits(server, leaf_uuid, instance, 'dvid', 'dict')
    
    bad_merge_events = []
    bad_cleave_events = []
    
    body_sizes = {}
    for cur_uuid in tqdm_proxy(audit_uuids):
        
        def get_body_size(body):
            if body in body_sizes:
                return body_sizes[body]
            try:
                body_sizes[body] = fetch_size(server, prev_uuid, instance, body)
                return body_sizes[body]
            except Exception:
                return 0
        
        logger.info(f"Auditing uuid '{cur_uuid}'")
        cur_msgs_df = msgs_df.query('uuid == @cur_uuid')
        cur_bad_merge_events = []
        cur_bad_cleave_events = []
        for row in tqdm_proxy(cur_msgs_df.itertuples(index=False), total=len(cur_msgs_df), leave=False):
            if row.target_body in ann_df.index:                
                target_status = ann_df.loc[row.target_body, 'status']
            else:
                target_status = None
            
            # Check for merges that eliminate a traced body
            if row.action == "merge":
                target_body_size = get_body_size(row.target_body)
                # Check for traced bodies being merged INTO another 
                _merged_labels = row.msg['Labels']
                bad_merges_df = ann_df.query('body in @_merged_labels and status in @TRACED_STATUSES')
                if len(bad_merges_df) > 0:
                    if target_status in TRACED_STATUSES:
                        cur_bad_merge_events.append(('merge-traced-to-traced', *row, target_status, bad_merges_df.index.tolist(), bad_merges_df['status'].tolist()))
                    else:
                        cur_bad_merge_events.append(('merge-traced-to-nontraced', *row, target_status, bad_merges_df.index.tolist(), bad_merges_df['status'].tolist()))

                # Update local body_sizes
                merged_size = sum([get_body_size(merge_label) for merge_label in row.msg['Labels']])
                if target_body_size != 0:
                    body_sizes[row.target_body] += merged_size
            
            # Check for bodies that lost more than half of their size in a cleave
            if row.action == "cleave" and target_status in TRACED_STATUSES:
                target_body_size = get_body_size(row.target_body)
                if target_body_size == 0:
                    # Since we aren't assessing split events yet,
                    # it's possible that we don't have all the information we need to assess all bodies.
                    # Bodies that were created during splits are not tracked.
                    cur_bad_cleave_events.append(( 'failed-to-assess', *row, target_body_size, 0 ))
                else:
                    cleaved_sizes = fetch_sizes(server, cur_uuid, instance, row.msg['CleavedSupervoxels'], supervoxels=True)
                    for sv in tqdm_proxy(cleaved_sizes[(cleaved_sizes == 0)].index, leave=False):
                        cleaved_sizes.loc[sv] = fetch_retired_supervoxel_size(server, leaf_uuid, instance, sv, split_events)
                    
                    if cleaved_sizes.sum() > target_body_size / 2:
                        cur_bad_cleave_events.append(( 'large-cleave', *row, target_body_size, cleaved_sizes.sum() ))
                    
                    # Update local body_sizes
                    body_sizes[row.target_body] -= cleaved_sizes.sum()
                    body_sizes[row.msg['CleavedLabel']] = cleaved_sizes.sum()
            
            # TODO: Check body split events, too.
            # We could apply the above message effects to ann_df,
            # but it doesn't actually matter.

        logger.info(f"Found {len(cur_bad_merge_events)} bad merges and {len(cur_bad_cleave_events)} bad cleaves")
        bad_merge_events += cur_bad_merge_events
        bad_cleave_events += cur_bad_cleave_events

        # Rather than fetching the entire ann_df for the next uuid,
        # just update the keys that changed (according to kafka).
        
        # TODO: It would be interesting to compare the difference between
        #       statuses that we computed vs. the statuses in the new uuid
        updated_bodies = ann_kv_log_df.query('uuid == @cur_uuid')['body']
        ann_update_df = fetch_body_annotations(server, cur_uuid, f'{instance}_annotations', updated_bodies)[['status']]
        ann_df = pd.concat((ann_df, ann_update_df))
        ann_df = ann_df.loc[~(ann_df.index.duplicated(keep='last'))]

    bad_merges_df = pd.DataFrame(bad_merge_events, columns=['reason', *msgs_df.columns, 'target_status', 'traced_bodies', 'traced_statuses'])
    bad_cleaves_df = pd.DataFrame(bad_cleave_events, columns=['reason', *msgs_df.columns, 'target_size', 'cleave_size'])
    
    np.save('audit_bad_merges_df.npy', bad_merges_df.to_records(index=False))
    np.save('audit_bad_cleaves_df.npy', bad_cleaves_df.to_records(index=False))
    
    return bad_merges_df, bad_cleaves_df


if __name__ == "__main__":
    from neuclease import configure_default_logging
    configure_default_logging()
    #audit_traced_history_for_branch('emdata4:8900', start_uuid='e939fd903fa24c46b79984cef120b04a')
    audit_traced_history_for_branch('emdata4:8900')
