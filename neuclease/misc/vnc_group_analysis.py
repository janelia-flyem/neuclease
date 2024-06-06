import os
import re
import copy
import json
import logging
import datetime
import warnings
from itertools import chain

import numpy as np
import pandas as pd
import networkx as nx
from bokeh.plotting import figure, output_file, save as bokeh_save, output_notebook, show
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Don't import these here -- use late imports, below.
#import holoviews as hv
#import hvplot.pandas
#import hvplot.networkx as hvnx

from networkx.classes.filters import show_nodes
from neuclease.misc.neuroglancer import format_nglink

from neuclease.util import fix_df_names, tqdm_proxy, Timer
from neuclease.dvid import find_master, fetch_body_annotations, post_keyvalues
from neuclease.clio.api import fetch_json_annotations_all

from neuclease import configure_default_logging

from .vnc_group_analysis_main import parse_args, VNC_PROD

logger = logging.getLogger(__name__)

DVID_TO_CLIO_SOMA_SIDE = {
    None: "",
    'L': 'LHS',
    'R': 'RHS',
    "M": "Midline",
    "UNP": "Midline",
    "U": "Midline"
}

CLIO_TO_DVID_SOMA_SIDE = {
    'LHS': 'L',
    'RHS': 'R',
    "Midline": 'M'
}


def main():
    args = parse_args()
    configure_default_logging()
    warnings.filterwarnings("ignore", message='.*figure.max_open_warning.*', category=RuntimeWarning)

    if args.uuid is None:
        args.uuid = find_master(VNC_PROD)

    if args.commit_dvid_updates:
        # Commit and exit
        clio_to_dvid_updates = pd.read_csv(args.commit_dvid_updates).set_index('body')
        commit_dvid_updates(args.dvid_server, args.uuid, clio_to_dvid_updates, dry_run=False)
        return

    vnc_group_conflict_report(args.dvid_server, args.uuid, None, None, args.plot_format, args.output_dir, args.skip_plots)


def vnc_group_conflict_report(server, uuid, dvid_ann=None, clio_ann=None, format='png', output_dir='clio-dvid-group-reports', skip_plots=False):
    """
    Download the 'group' annotations from clio and the 'instance' annotations from DVID,
    and determine which annotations conflict, i.e. some bodies seem to belong to different groups in Clio vs. DVID.

    Note that merely having different group names (group IDs) is not a conflict.
    We assess the topology of the group memberships and emit a report in HTML form,
    showing schematics of which bodies are tagged with which groups.
    """
    with Timer("Fetching DVID annotations", logger):
        dvid_ann = fetch_sanitized_dvid_annotations(server, uuid, dvid_ann)

    if clio_ann is None:
        with Timer("Fetching Clio annotations", logger):
            clio_ann = fetch_sanitized_clio_group_annotations()

    logger.info("Merging annotations")
    ann = merge_annotations(dvid_ann, clio_ann)

    logger.info("Computing union groups")
    g, group_df = compute_union_groups(ann)

    logger.info(f"Exporting report to {output_dir}")
    conflict_df, nonconflict_ann = export_report(g, group_df, ann, output_dir, format, skip_plots)

    num_conflict_bodies = conflict_df['num_bodies'].sum()
    logger.info(f"Found {len(conflict_df)} conflict sets, encompassing {num_conflict_bodies}  bodies.")

    dvid_to_clio_updates, clio_to_dvid_updates = determine_nonconflict_updates(ann, nonconflict_ann)
    logger.info(f"Non-conflicting updates that could be made from DVID -> Clio: {len(dvid_to_clio_updates)}")
    logger.info(f"Non-conflicting updates that could be made from Clio -> DVID: {len(clio_to_dvid_updates)}")
    logger.info("See CSV outputs.")

    dvid_to_clio_updates.to_csv(f"{output_dir}/dvid-to-clio-updates.csv", index=True, header=True)
    clio_to_dvid_updates.to_csv(f"{output_dir}/clio-to-dvid-updates.csv", index=True, header=True)

    return ann, conflict_df, nonconflict_ann, dvid_to_clio_updates, clio_to_dvid_updates


def commit_dvid_updates(server, uuid, clio_to_dvid_updates, dry_run=False):
    """
    Given the clio->dvid updates produced by the function above,
    actually commit those updates to DVID's segmentation_annotations instance.
    """
    assert clio_to_dvid_updates.index.name == 'body'
    assert {*clio_to_dvid_updates.columns} >= {'instance', 'instance_user'}

    logger.info("Fetching segmentation_annotations")
    ann = fetch_body_annotations(server, uuid, status_categories=None)
    ann = ann.merge(clio_to_dvid_updates, 'right', on='body', suffixes=['_orig', '_new']).fillna(False)
    ann.index = ann.index.astype(np.uint64)

    logger.info("Updating json")
    updates = {}
    update_ann = ann.query('instance_orig != instance_new')
    for row in update_ann.itertuples():
        j = row.json or {}
        j['instance'] = row.instance_new
        j['instance_user'] = row.instance_user_new
        body = row.Index
        updates[body] = j

    if not dry_run:
        logger.info(f"Posting {len(updates)} updates to segmentation_annotations")
        post_keyvalues(server, uuid, 'segmentation_annotations', updates)

    return updates


def determine_nonconflict_updates(ann, nonconflict_ann):
    """
    Given (sanitized) full annotations from clio and dvid, and the subset of those which are known not to "conflict",
    Produce mappings to indicate how the 'groups' should be updated in DVID and updated in Clio.

    Note that this isn't as simple as just copying the group ID from one database to the other,
    since identical groups might have different IDs in the two databases.
    First we have to establish the group ID translation mapping, then apply it to each side.

    Then we have to translate column names and terminology between DVID and Clio.
    """
    # Get the translation between DVID and Clio group IDs.
    # (Usually they'll be the same, but in some cases they may differ.)
    q = 'not group_dvid.isnull() and not group_clio.isnull()'
    mapping_df = ann.query(q)[['group_dvid', 'group_clio']].astype(int).drop_duplicates()
    dvid_groups = nonconflict_ann["group_dvid"].unique()  # noqa
    clio_groups = nonconflict_ann['group_clio'].unique()  # noqa
    mapping_df = mapping_df.query('group_dvid in @dvid_groups or group_clio in @clio_groups')
    assert (mapping_df['group_dvid'].value_counts() == 1).all()
    assert (mapping_df['group_clio'].value_counts() == 1).all()

    dvid_to_clio_updates = _determine_dvid_to_clio_updates(nonconflict_ann, mapping_df)
    clio_to_dvid_updates = _determine_clio_to_dvid_updates(nonconflict_ann, mapping_df)
    return dvid_to_clio_updates, clio_to_dvid_updates


def _determine_dvid_to_clio_updates(nonconflict_ann, mapping_df):
    # Replace 'group_clio' column using mapping from dvid->clio
    # ...and in places where there is nothing in the mapping yet, just copy directly from DVID
    dvid_to_clio_updates = (
        nonconflict_ann
            .reset_index()
            .merge(mapping_df, 'left', on='group_dvid', suffixes=['_orig', ''])
            .set_index('body')
            [['group_dvid', 'group_clio']]
            .ffill(axis=1)
            .query('not group_clio.isnull()')
            .astype(int)
    )

    # Now incorporate 'soma side' and user
    _df = nonconflict_ann[['group_clio', 'soma_side_dvid', 'user_dvid']]
    dvid_to_clio_updates = dvid_to_clio_updates.merge(_df, 'inner', on='body', suffixes=['', '_orig'])

    # Filter out non-changes
    dvid_to_clio_updates = dvid_to_clio_updates.query('group_clio != group_clio_orig').copy()

    # Select/rename columns
    renames = {
        'group_clio': 'group',
        'soma_side_dvid': 'soma_side',
        'user_dvid': 'naming_user'
    }
    dvid_to_clio_updates = dvid_to_clio_updates.rename(columns=renames)
    dvid_to_clio_updates = dvid_to_clio_updates[['group', 'soma_side', 'naming_user']]
    return dvid_to_clio_updates


def _determine_clio_to_dvid_updates(nonconflict_ann, mapping_df):
    # Replace 'group_dvid' column using mapping from clio->dvid
    # ...and in places where there is nothing in the mapping yet, just copy directly from Clio
    clio_to_dvid_updates = (
        nonconflict_ann
            .reset_index()
            .merge(mapping_df, 'left', on='group_clio', suffixes=['_orig', ''])
            .set_index('body')
            [['group_dvid', 'group_clio']]
            .bfill(axis=1)
            .query('not group_dvid.isnull()')
            .astype(int)
    )

    # Now incorporate 'soma side'
    _df = nonconflict_ann[['group_dvid', 'soma_side_clio', 'user_clio']]
    clio_to_dvid_updates = clio_to_dvid_updates.merge(_df, 'inner', on='body', suffixes=['', '_orig'])

    # Filter out non-changes
    clio_to_dvid_updates = clio_to_dvid_updates.query('group_dvid != group_dvid_orig').copy()

    # Construct 'instance' column
    def _construct_instance(row):
        if row.soma_side_code:
            return f"{row.group_dvid}_{row.soma_side_code}"
        else:
            return str(row.group_dvid)

    clio_to_dvid_updates['soma_side_code'] = clio_to_dvid_updates['soma_side_clio'].map(lambda s: CLIO_TO_DVID_SOMA_SIDE.get(s, s))
    clio_to_dvid_updates['instance'] = [*map(_construct_instance, clio_to_dvid_updates.itertuples())]

    # Select/rename columns
    clio_to_dvid_updates = clio_to_dvid_updates.rename(columns={'user_clio': 'instance_user'})
    clio_to_dvid_updates = clio_to_dvid_updates[['instance', 'instance_user']]
    return clio_to_dvid_updates


def fetch_sanitized_dvid_annotations(server, uuid, dvid_ann=None):
    """
    Fetch from segmentation_annotations, but only those containing an instance.

    - split 'instance' into group (int) and soma side (string, using "" where missing)
    - rename column: 'instance_user' -> 'user'
    - Convert 'soma_side' to clio terminology (LHS, RHS, Midline)
    """
    if dvid_ann is None:
        dvid_ann = fetch_body_annotations(server, uuid)
    dvid_ann = fix_df_names(dvid_ann)
    dvid_ann = dvid_ann.query('not instance.isnull() and instance != ""').copy()
    dvid_ann['instance'] = dvid_ann['instance'].map(str)

    # Possible instances:
    # "1234"
    # "1234_L"
    # "1234_R"
    # "1234_M"
    # "1234_UNP"
    # "1234R"
    # "(1234_L)"
    pat = re.compile(r'\(?([0-9]+)_?([A-Z]+)?\)?')

    matches = dvid_ann['instance'].map(pat.match)
    if matches.isnull().any():
        msg = "Can't parse instance for some bodies\n"
        msg += str(dvid_ann.loc[matches.isnull(), 'instance'].to_frame().reset_index())
        msg += "\n"
        msg += "Ignoring those bodies."
        logger.warn(msg)

        dvid_ann = dvid_ann.loc[~matches.isnull()].copy()
        matches = matches.loc[~matches.isnull()]

    dvid_ann['group'] = matches.map(lambda m: int(m.groups()[0]))
    dvid_ann['soma_side'] = matches.map(lambda m: m.groups()[1])
    dvid_ann['soma_side'] = dvid_ann['soma_side'].map(lambda s: DVID_TO_CLIO_SOMA_SIDE.get(s, s))
    dvid_ann = dvid_ann[['instance', 'group', 'soma_side', 'instance_user']].rename(columns={'instance_user': 'user'})
    return dvid_ann


def fetch_sanitized_clio_group_annotations(clio_ann=None):
    """
    Fetch clio body annotations, but only those containing
    either a 'group' or soma_side (or both).

    - rename 'bodyid' to 'body' and use it as the index
    """
    if clio_ann is None:
        clio_ann = fetch_json_annotations_all('VNC', 'neurons', 'pandas')

    clio_ann = clio_ann.set_index('bodyid').rename_axis('body')
    clio_ann = clio_ann.query('(not group.isnull() and group != "") or (not soma_side.isnull() and soma_side != "")').copy()
    clio_ann['group'] = clio_ann['group'].map(lambda g: np.nan if g == "" else float(g))
    clio_ann['soma_side'] = clio_ann['soma_side'].fillna("")
    clio_ann['soma_side'] = clio_ann['soma_side'].map(lambda s: {'RHs': 'RHS', 'None': "", "TBD": ""}.get(s, s))
    return clio_ann


def merge_annotations(dvid_ann, clio_ann):
    """
    Merge dvid and clio annotations (each of which must have been pre-sanitized).
    Groups will be converted to unique strings, for convenient insertion into a nx.Graph later on.
    """
    ann = dvid_ann.merge(clio_ann[['group', 'soma_side', 'user']], 'outer', left_index=True, right_index=True, suffixes=['_dvid', '_clio'])

    # Convert groups to strings for graph analysis
    ann = ann.query('group_clio != ""').copy()

    ann['group_dvid_name'] = ""
    ann['group_clio_name'] = ""

    idx = ~(ann['group_dvid'].isnull())
    dvid_names = 'dvid_' + ann.loc[idx, 'group_dvid'].astype(int).astype(str)
    ann.loc[idx, 'group_dvid_name'] = dvid_names

    idx = ~(ann['group_clio'].isnull())
    clio_names = 'clio_' + ann.loc[~(ann['group_clio'].isnull()), 'group_clio'].astype(int).astype(str)
    ann.loc[idx, 'group_clio_name'] = clio_names

    return ann


def get_soma_side_disagreements(ann):
    # Soma side disagreements
    q = ('not soma_side_dvid.isnull() and not soma_side_clio.isnull()'
        ' and soma_side_dvid != "" and soma_side_clio != ""'
        ' and soma_side_dvid != soma_side_clio')
    soma_disagreements = ann.query(q)
    return soma_disagreements


def compute_union_groups(ann):
    # Construct group graph
    g = nx.Graph()
    g.add_edges_from(ann.query('group_dvid_name != ""').reset_index()[['body', 'group_dvid_name']].values)
    g.add_edges_from(ann.query('group_clio_name != ""').reset_index()[['body', 'group_clio_name']].values)

    union_groups = []
    for cc in nx.connected_components(g):
        bodies = []
        dvid_groups = []
        clio_groups = []
        for node in cc:
            if isinstance(node, int):
                bodies.append(node)
            elif node.startswith('dvid_'):
                dvid_groups.append(int(float(node[len('dvid_'):])))
            elif node.startswith('clio_'):
                clio_groups.append(int(float(node[len('clio_'):])))
            else:
                assert False
        union_groups.append((len(bodies), len(dvid_groups), len(clio_groups), bodies, dvid_groups, clio_groups, cc))

    df = pd.DataFrame(union_groups, columns=['num_bodies', 'num_dvid_groups', 'num_clio_groups', 'bodies', 'dvid_groups', 'clio_groups', 'cc'])
    df = df.sort_values(['num_dvid_groups', 'num_clio_groups', 'num_bodies'], ascending=False).reset_index(drop=True)
    return g, df


def plot_union_group(g, df, idx, format='hv'):
    assert format in ('hv', 'png')
    nodes = list(df.loc[idx, 'cc'])
    sg = nx.subgraph_view(g, show_nodes(nodes))

    colors = []
    for n in sg.nodes():
        if isinstance(n, int):
            colors.append('#cccccc')
        elif n.startswith('dvid'):
            #colors.append('#87CEEB')
            dvid_group = int(n[len('dvid_'):])
            colors.append(pick_color(dvid_group))
        elif n.startswith('clio'):
            #colors.append('#00FF7F')
            clio_group = int(n[len('clio_'):])
            colors.append(pick_color(clio_group))
        else:
            assert False

    #pos = nx.spring_layout(sg)
    pos = nx.kamada_kawai_layout(sg)
    OFFSET = 0.02
    label_pos = {n: (x, y+OFFSET) for n, (x,y) in pos.items()}

    if format == 'hv':
        import hvplot.networkx as hvnx
        pn = hvnx.draw(sg, pos=pos, node_color=colors)
        pl = hvnx.draw_networkx_labels(sg, label_pos)
        p = pn * pl
        p = p.opts(height=800, width=1000, title=f"conflict set #{idx}")
        return p
    else:
        fig, ax = plt.subplots()
        dpi = 96
        fig.set_dpi(dpi)
        fig.set_size_inches(1300/dpi, 1000/dpi)
        ax.set_title(f"conflict set #{idx}", fontsize=30)
        nx.draw(sg, pos=pos, node_color=colors, ax=ax) # edgecolors='black',
        nx.draw_networkx_labels(sg, label_pos, ax=ax)
        return fig


def pick_color(i):
    if i == 0:
        return "#ffffff"
    hue = ((hash(str(i)) // 7) % 100) / 100
    rgb = hsv_to_rgb([hue, 1, 1])
    r, g, b = (rgb * 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def export_report(g, group_df, ann, output_dir, format='hv', skip_plots=False):
    # Late imports because these imports can conflict with bokeh server apps
    import holoviews as hv
    import hvplot.pandas

    os.makedirs(output_dir, exist_ok=True)

    today = datetime.datetime.today().strftime("%Y-%m-%d")

    non_conflict_bodies = [*chain(*group_df.query('num_dvid_groups <= 1 and num_clio_groups <= 1')['bodies'])]
    nonconflict_ann = ann.loc[non_conflict_bodies]
    nonconflict_ann.to_csv(f'{output_dir}/nonconflicting-annotations-sanitized.csv', index=True, header=True)

    conflict_df = group_df.query('num_dvid_groups > 1 or num_clio_groups > 1').copy()
    conflict_df = conflict_df.rename_axis('conflict_set')
    conflict_df.index += 1

    #conflict_df = conflict_df.iloc[:10]

    _construct_links(conflict_df, ann)

    csv_path = f'{output_dir}/conflict-report-{today}.csv'
    conflict_df.to_csv(csv_path, index=True, header=True)

    if skip_plots:
        return conflict_df, nonconflict_ann

    conflict_df['num_dvid_groups'] *= -1
    summary_bars = conflict_df[['num_dvid_groups', 'num_clio_groups']].iloc[::-1].hvplot.barh(
        stacked=True, title='unioned group counts', legend='bottom_right', height=12 * len(conflict_df)).opts(xlabel='union id', ylabel='number')

    output_file(filename=f"{output_dir}/summary.html", title=f"Conflict Set Sizes {today}")
    bokeh_save(hv.render(summary_bars))
    conflict_df['num_dvid_groups'] *= -1

    # Construct plots
    plots = []
    for idx in tqdm_proxy(conflict_df.index):
        p = plot_union_group(g, conflict_df, idx, format)
        plots.append(p)

    # Export plots
    # Note: the 'png' option also outputs links and body lists into the HTML
    assert format in ('hv', 'png')
    if format == 'hv':
        layout = hv.Layout(plots).cols(1)
        output_file(filename=f"{output_dir}/conflict-groups.html", title=f"Conflict Report {today}")
        bokeh_save(hv.render(layout))
    elif format == 'png':
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        with open(f"{output_dir}/conflict-groups.html", 'w') as f:
            f.write("<html>\n")
            f.write(f"<head><title>Conflict Report {today}</title></head>\n")
            f.write("<body>\n")
            for idx, p in zip(tqdm_proxy(conflict_df.index), plots):
                p.savefig(f'{output_dir}/plots/{idx}.png')
                f.write(f"<h1>Conflict set #{idx}</h1>")

                num_bodies, num_dvid_groups, num_clio_groups = conflict_df.loc[idx, ['num_bodies', 'num_dvid_groups', 'num_clio_groups']]
                f.write(f"{num_dvid_groups} DVID groups<br>\n")
                f.write(f"{num_clio_groups} clio groups<br>\n")
                f.write(f"{num_bodies} bodies<br>\n")

                bodies = ', '.join(map(str, conflict_df.loc[idx, 'bodies']))
                f.write(bodies)
                f.write('<br>\n')

                link = conflict_df.loc[idx, 'link']
                f.write(f'<a href="{link}">Comparison Link #{idx}</a><br>\n')

                f.write(f"<img src=plots/{idx}.png><br>\n")
            f.write("</body>\n")
            f.write("</html>\n")

    return conflict_df, nonconflict_ann


def _construct_links(group_df, ann):
    """
    Add a 'link' column to group_df which adds two color-coded segmentation layers (for dvid and clio groups).
    """
    link_settings_path = os.path.split(__file__)[0] + '/conflict-comparison-settings.json'
    link_settings = json.load(open(link_settings_path, 'r'))

    group_df['link'] = ""
    for idx in group_df.index:
        link = copy.deepcopy(link_settings)
        link['title'] = f"Comparison #{idx}"
        seg_layer = [l for l in link['layers'] if l['name'].startswith('seg')][0]
        dvid_layer = [l for l in link['layers'] if l['name'] == "dvid-groups"][0]
        clio_layer = [l for l in link['layers'] if l['name'] == "clio-groups"][0]

        bodies = group_df.loc[idx, 'bodies']
        seg_layer['segments'] = [*map(str, bodies)]
        dvid_layer['segments'] = seg_layer['segments']
        clio_layer['segments'] = seg_layer['segments']

        seg_layer['segmentQuery'] = ', '.join(seg_layer['segments'])
        dvid_layer['segmentQuery'] = seg_layer['segmentQuery']
        clio_layer['segmentQuery'] = seg_layer['segmentQuery']

        _df = ann.loc[bodies, ['group_dvid', 'group_clio']].fillna(0).astype(int)
        dvid_layer['segmentColors'] = {str(b): pick_color(g) for b, g in _df['group_dvid'].items()}
        clio_layer['segmentColors'] = {str(b): pick_color(g) for b, g in _df['group_clio'].items()}

        group_df.loc[idx, 'link'] = format_nglink('https://clio-ng.janelia.org', link)


if __name__ == "__main__":
    main()
