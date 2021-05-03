"""
For every body with a soma, upgrade its body annotation to
'Soma Anchor' unless it already has that status or better.

Example:
    python neuclease/misc/tag_soma_anchors.py emdata5.janelia.org:8400 master soma-bookmarks
"""
import copy
import logging

logger = logging.getLogger(__name__)


def tag_soma_anchors(server, uuid, soma_instance, seg_instance="segmentation", ann_instance=None, dry_run=False):
    """
    For every body with a soma, upgrade its body annotation to
    'Soma Anchor' unless it already has that status or better.

    Args:
        server:
            dvid server
        uuid:
            uuid to read and write to.
            The string "master" can be used as a shortcut for the most recent node on the master branch.
        soma_instance:
            keyvalue instance containing soma line annotations
        seg_instance:
            labelmap instance
        ann_instance:
            body annotation keyvalue instance
        dry_run:
            If True, determine which bodies to upgrade, but don't write the updates.

    """
    from neuclease.dvid import find_master, fetch_sphere_annotations, fetch_body_annotations, post_keyvalues, DEFAULT_BODY_STATUS_CATEGORIES

    if uuid == "master":
        uuid = find_master(server)

    if ann_instance is None:
        ann_instance = f"{seg_instance}_annotations"

    logger.info("Fetching soma annotations")
    soma_df = fetch_sphere_annotations(server, uuid, soma_instance, seg_instance)

    logger.info("Fetching body annotations")
    ann_df = fetch_body_annotations(server, uuid, ann_instance, soma_df['body'])

    soma_df = soma_df.query('body != 0')
    soma_df = soma_df.merge(ann_df['status'], 'left', left_on='body', right_index=True)
    soma_df['status'].fillna("", inplace=True)

    all_statuses = DEFAULT_BODY_STATUS_CATEGORIES
    keep_statuses = all_statuses[all_statuses.index('Soma Anchor'):]
    upgrade_df = soma_df.query('status not in @keep_statuses')

    new_data = {}
    for body in upgrade_df['body'].tolist():
        try:
            body_dict = copy.copy(ann_df.loc[body, 'json'])
        except KeyError:
            body_dict = {"body ID": body}

        body_dict["status"] = "Soma Anchor"
        new_data[str(body)] = body_dict

    status_counts = upgrade_df['status'].value_counts().rename("count").rename_axis("status").to_frame().query("count > 0")
    logger.info(f"Upgrading {len(upgrade_df)} statuses from:\n" + str(status_counts))
    logger.info(f"Bodies: {' '.join(new_data.keys())}")

    if not dry_run:
        post_keyvalues(server, uuid, ann_instance, new_data)
    logger.info("DONE")

    upgrade_df[['body', 'status', *'xyz']].to_csv('upgraded-soma-bodies.csv', header=True, index=False)
    return upgrade_df

if __name__ == "__main__":
    import argparse
    from neuclease import configure_default_logging
    configure_default_logging()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument("server")
    parser.add_argument("uuid")
    parser.add_argument("soma_instance")
    parser.add_argument("seg_instance", nargs='?', default='segmentation')
    parser.add_argument("ann_instance", nargs='?')
    args = parser.parse_args()

    tag_soma_anchors(args.server, args.uuid, args.soma_instance, args.seg_instance, args.ann_instance, args.dry_run)
