"""
Create a set of branches for a list of users (one branch per user),
and configure the special 'branches' keyvalue instance for NeuTu's
UUID alias feature.

Example usage:

    NOTE='Proofreading exercises for {user}'
    prepare_user_branches \\
        -b prtech_20211019 \\
        -n "${NOTE}" \\
        --use-custom-uuids \\
        emdata4.int.janelia.org:9300 \\
        5a7d0c59a918400181aaac6144f4ede2 \\
        prtech \\
        users.csv

---
"""
import os
import argparse
import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dvid_server")
    parser.add_argument(
        "parent_uuid",
        help="UUID underneath which all branches will be created")
    parser.add_argument(
        "neutu_alias_name",
        help='The alias that users will enter in the NeuTu settings window, e.g. "prtech"')
    parser.add_argument(
        "usernames_csv",
        help='CSV file (just one column) of usernames for which branches will be created (one branch per user)')
    parser.add_argument(
        "--branch-prefix", '-b',
        help=('A prefix to be used when naming the branches (and UUIDs). '
              'If not provided, the neutu_alias_name is used. '
              '(Hint: If you plan to re-use the alias in the future, then use a custom '
              'prefix, e.g. "prtech_20211019")'))
    parser.add_argument(
        "--note", '-n',
        help=('The note that will appear in the DVID console. '
              'As a special feature, you may include "{user}" in the string, which will '
              'be replaced with the username corresponding to the branch."'))
    parser.add_argument(
        "--use-custom-uuids", '-u', action='store_true',
        help=("If True, don't let DVID choose a random UUID. "
              "Instead, force a custom UUID, which will match the branch name."))

    args = parser.parse_args()

    from neuclease import configure_default_logging
    configure_default_logging()

    import pandas as pd
    usernames = pd.read_csv(args.usernames_csv, header=None, names=['user'])['user']

    logger.info(f"Creating/configuring branches for {len(usernames)} users.")

    prepare_user_branches(
        args.dvid_server,
        args.parent_uuid,
        args.neutu_alias_name,
        usernames,
        args.branch_prefix,
        args.note,
        args.use_custom_uuids
    )

    logger.info("DONE")


def prepare_user_branches(server, parent_uuid, neutu_alias_name, usernames, branch_prefix=None, uuid_note=None, use_custom_uuids=False):
    """
    Create a set of branches for a list of users (one branch per user),
    and configure the special 'branches' keyvalue instance for NeuTu's
    UUID alias feature.

    Args:
        server:
            DVID server
        parent_uuid:
            UUID underneath which all branches will be created
        neutu_alias_name:
            The alias that users will enter in the NeuTu settings window, e.g. "prtech"
        usernames:
            List of usernames for which branches will be created (one branch per user)
        branch_prefix:
            Optional. A prefix to be used when naming the branches (and UUIDs).
            If not provided, the neutu_alias_name is used.
            Hint: If you plan to re-use the alias in the future, then use a custom
            prefix, e.g. "prtech_20211019"
        uuid_note:
            The 'note' that will appear in the DVID console.
            As a special feature, you may include "{user}" in the string, which will
            be replaced with the username corresponding to the branch.
        use_custom_uuids:
            If True, don't let DVID choose a random UUID.  Instead, force a custom UUID,
            which will match the branch name.
    """
    from neuclease.dvid import post_branch, post_key, find_repo_root

    if not os.environ.get('DVID_ADMIN_TOKEN'):
        msg = ("This script will need to write to the DVID repo root node, "
               "so you'll need to define DVID_ADMIN_TOKEN in your environment.")
        raise RuntimeError(msg)

    root_uuid = find_repo_root(server, parent_uuid)
    uuid_prefix = branch_prefix or f"{neutu_alias_name}_"
    uuid_note = uuid_note or f"{uuid_prefix} branch for {{user}}"

    for user in usernames:
        branch = f"{uuid_prefix}_{user}"
        if use_custom_uuids:
            custom_uuid = branch

        logger.info(f"Configuring branch {branch} in uuid {custom_uuid}")
        note = uuid_note.format(user=user)
        uuid = post_branch(server, parent_uuid, branch, note, custom_uuid=custom_uuid)
        post_key(server, root_uuid, 'branches', f"{neutu_alias_name}_{user}", uuid.encode('utf-8'))


if __name__ == "__main__":
    main()
