import logging
import getpass
from functools import partial
import pandas as pd

from neuclease.util import compute_parallel
from neuclease.dvid import (fetch_body_annotations, fetch_sphere_annotations, fetch_all_elements, fetch_labels_batched,
                            post_keyvalues, fetch_counts, fetch_sparsevol_head, delete_key)

logger = logging.getLogger(__name__)


def fetch_vnc_statuses(server, uuid):
    """
    Fetch all body statuses from the body annotation key-value,
    but also include all soma bodies (regardless of status)
    and bodies that were annotated in the neck.

    Also fetch the number of synapses for each.

    Example:

        .. code-block:: ipython

            In [72]: ann = fetch_vnc_statuses('emdata5.janelia.org:8400', '73f39bea795f48e18feafb033b544ae5')
            [2021-05-06 11:32:14,581] INFO Pre-sorting 15143 coordinates by block index...
            [2021-05-06 11:32:14,587] INFO Pre-sorting 15143 coordinates by block index took 0:00:00.006287
            [2021-05-06 11:32:14,588] INFO Fetching labels from DVID...
            [2021-05-06 11:32:26,116] INFO Fetching labels from DVID took 0:00:11.527091
            [2021-05-06 11:32:31,480] WARNING There are 129 duplicate bodies in the results, due to multi-soma and/or multi-cervical bodies!

            In [73]: ann.columns
            Out[73]:
            Index(['status', 'user', 'naming user', 'instance', 'status user', 'comment',
                'json', 'soma_x', 'soma_y', 'soma_z', 'has_soma', 'neck_x', 'neck_y',
                'neck_z', 'is_cervical'],
                dtype='object')

            In [75]: ann.query('has_soma or is_cervical')[['status', 'status user', 'has_soma', 'is_cervical',
                ...:                                       'soma_x', 'soma_y', 'soma_z', 'neck_x', 'neck_y', 'neck_z']]
            Out[75]:
                                   status  status user has_soma  is_cervical  soma_x  soma_y  soma_z  neck_x  neck_y  neck_z
            body
            10000   Prelim Roughly traced                 False         True       0       0       0   24481   36044   67070
            100000            Soma Anchor                  True        False   22959   20811    7254       0       0       0
            100002            Soma Anchor                  True        False   28216   35641   61443       0       0       0
            10002   Prelim Roughly traced                 False         True       0       0       0   23217   35252   67070
            100031  Prelim Roughly traced      smithc     False         True       0       0       0   23263   38354   67070
            ...                       ...         ...       ...          ...     ...     ...     ...     ...     ...     ...
            97550         Cervical Anchor                 False         True       0       0       0   23341   38451   67070
            99837   Prelim Roughly traced       cookm     False         True       0       0       0   22665   38397   67070
            0                                              True        False   14912   31188   19347       0       0       0
            0                                              True        False   23125   16634   12777       0       0       0
            167778                                         True        False   22324    6881   16642       0       0       0

            [17188 rows x 10 columns]

    """
    soma_df = fetch_sphere_annotations(server, uuid, 'soma-bookmarks', 'segmentation')
    soma_df = soma_df[['body', *'xyz']]
    soma_df['has_soma'] = True
    soma_df = soma_df.rename(columns={k: f'soma_{k}' for k in 'xyz'})

    neck_df = fetch_all_elements(server, uuid, 'neck-points', format='pandas')
    neck_df = neck_df[[*'xyz']]
    neck_df['body'] = fetch_labels_batched(server, uuid, 'segmentation', neck_df[[*'zyx']].values, processes=4, batch_size=1000)
    neck_df = neck_df.rename(columns={k: f'neck_{k}' for k in 'xyz'})
    neck_df['is_cervical'] = True

    ann_df = fetch_body_annotations(server, uuid, 'segmentation_annotations')
    ann_df = ann_df.reset_index()
    ann_df = ann_df.merge(soma_df, 'outer', on='body')
    ann_df = ann_df.merge(neck_df, 'outer', on='body')

    ann_df['has_soma'].fillna(False, inplace=True)
    ann_df['is_cervical'].fillna(False, inplace=True)

    for c in ann_df.columns:
        if c[-2:] in ('_x', '_y', '_z'):
            ann_df[c] = ann_df[c].fillna(0).astype(int)

    for c in ('status', 'user', 'naming user', 'instance', 'status user', 'comment'):
        ann_df[c].fillna("", inplace=True)

    dupes = ann_df['body'].duplicated().sum()
    if dupes:
        logger.warn(f"There are {dupes} duplicate bodies in the results, due to multi-soma and/or multi-cervical bodies!")

    del ann_df['body ID']

    ann_df = ann_df.set_index('body')

    bodies = ann_df.index.drop_duplicates().values
    ann_df['tbars'] = fetch_counts(server, uuid, 'synapses_labelsz', bodies, 'PreSyn', format='pandas')
    ann_df['psds'] = fetch_counts(server, uuid, 'synapses_labelsz', bodies, 'PostSyn', format='pandas')
    ann_df['synapses'] = ann_df.eval('tbars + psds')

    return ann_df


def multisoma(ann):
    """
    Given the status table as returned by fetch_vnc_statuses (above),
    select only the rows for multi-soma bodies, i.e. bodies which
    cover more than one soma point annotation.

    Note: We exclude body 0, even though there might be soma annotations on voxels with label 0.
    """
    soma_df = ann.query('has_soma and body != 0').copy()
    idx = soma_df.groupby('body')['soma_x'].nunique() > 1
    multisoma_df = soma_df.loc[idx].sort_index().drop_duplicates(['soma_x', 'soma_y', 'soma_z'])
    return multisoma_df


def multicervical(ann):
    """
    Given the status table as returned by fetch_vnc_statuses (above),
    select only the rows for multi-cervical bodies, i.e. bodies which
    cover more than one neck point annotation.
    """
    cervical_df = ann.query('is_cervical').copy()
    idx = cervical_df.groupby('body')['neck_x'].nunique() > 1
    multicervical_df = cervical_df.loc[idx].sort_index().drop_duplicates(['neck_x', 'neck_y', 'neck_z'])
    return multicervical_df


def update_soma_statuses(server, uuid, dry_run=True):
    vstats = fetch_vnc_statuses(server, uuid)

    soma_statuses = {"Prelim Roughly traced", "Soma Anchor"}  # noqa
    needs_upgrade = vstats.query('has_soma and status not in @soma_statuses').copy()
    needs_downgrade = vstats.query('not has_soma and status == "Soma Anchor"').copy()

    print("\n-------------------\n")

    if (len(needs_downgrade) == 0 and len(needs_upgrade) == 0):
        print("Found nothing to upgrade/downgrade.")
        return

    if len(needs_upgrade) > 0:
        print(f"{len(needs_upgrade)} bodies need to be upgraded to Soma Anchor:")
        vc = needs_upgrade['status'].value_counts()
        print(vc[vc > 0].rename('count').rename_axis('status').to_frame())
        print()
        print(', '.join(map(str, needs_upgrade.index.tolist())))
        print()
        print(needs_upgrade[['status', 'has_soma', 'soma_x', 'soma_y', 'soma_z', 'is_cervical', 'status user']])
        print("\n-------------------\n")

    if len(needs_downgrade) > 0:
        print(f"{len(needs_downgrade)} bodies need to be downgraded to Anchor:")
        vc = needs_downgrade['status'].value_counts()
        print(vc[vc > 0].rename('count').rename_axis('status').to_frame())
        print()
        print(', '.join(map(str, needs_downgrade.index.tolist())))
        print()
        print(needs_downgrade[['status', 'has_soma', 'soma_x', 'soma_y', 'soma_z', 'is_cervical', 'status user']])
        print("\n-------------------\n")

    if dry_run:
        print("Dry run. Doing nothing.")
        return

    if len(needs_upgrade) > 0:
        needs_upgrade['json'] = needs_upgrade['json'].map(lambda j: j if isinstance(j, dict) else {})
        assert needs_upgrade['json'].map(lambda j: isinstance(j, dict)).all()
        for body, j in needs_upgrade['json'].items():
            if not isinstance(j, dict):
                j = {}

            j.update({
                'body ID': int(body),
                'status': 'Soma Anchor',
                'user': getpass.getuser(),
                'status user': getpass.getuser()
            })

        print(f"Upgrading {len(needs_upgrade)} statuses")
        post_keyvalues(server, uuid, 'segmentation_annotations', dict(needs_upgrade['json'].items()))

    if len(needs_downgrade) > 0:
        needs_downgrade['json'] = needs_downgrade['json'].map(lambda j: j if isinstance(j, dict) else {})
        assert needs_downgrade['json'].map(lambda j: isinstance(j, dict)).all()
        for body, j in needs_downgrade['json'].items():
            if not isinstance(j, dict):
                j = {}

            j.update({
                'body ID': int(body),
                'status': 'Anchor',
                'user': getpass.getuser(),
                'status user': getpass.getuser()
            })

        print(f"Downgrading {len(needs_downgrade)} statuses")
        post_keyvalues(server, uuid, 'segmentation_annotations', dict(needs_downgrade['json'].items()))

    print("Done")


def post_statuses(server, uuid, statuses):
    assert isinstance(statuses, pd.Series)
    assert statuses.index.name == 'body'
    assert statuses.name == 'status'

    ann = fetch_body_annotations(server, uuid, bodies=statuses.index)
    ann = ann.merge(statuses.rename('new_status'), 'right', left_index=True, right_index=True)

    to_change = ann.query('status.isnull() or status != new_status')
    updates = {}
    for row in to_change.itertuples():
        j = row.json
        if not isinstance(row.json, dict):
            j = {}

        j.update({
            'body ID': int(row.Index),
            'status': row.new_status,
            'user': getpass.getuser(),
            'status user': getpass.getuser()
        })
        updates[row.Index] = j

    if len(to_change) == 0:
        print("Nothing to change")
        return
    
    print(f"Changing {len(to_change)} statuses")
    post_keyvalues(server, uuid, 'segmentation_annotations', updates)


def remove_dead_annotations(server, uuid):
    ann = fetch_body_annotations(server, uuid)

    exists = compute_parallel(partial(_does_exist, server, uuid), ann.index.values, processes=16)
    exists = pd.DataFrame(exists, columns=['body', 'exists'])

    to_del = exists.query('not exists')['body']
    print(f"deleting {len(to_del)} dead annotations: {to_del.tolist()}")
    for body in to_del.values:
        delete_key(server, uuid, 'segmentation_annotations', str(body))


def _does_exist(server, uuid, body):
    return (body, fetch_sparsevol_head(server, uuid, 'segmentation', body))
