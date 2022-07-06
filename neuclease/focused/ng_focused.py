from neuclease.util.util import unsplit_json_int_lists
import os
import json

import numpy as np
import pandas as pd

from neuclease.util import iter_batches, convert_nans, dump_json


ASSIGNMENT_EXAMPLE = """\
{
    "file type": "Neu3 task list",
    "file version": 1,
    "grayscale source": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg",
    "segmentation source": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.1/segmentation",
    "task set description": "example task set",
    "task list": [
      {
        "task type": "body merge",
        "supervoxel ID 1": 1684779791,
        "supervoxel point 1": [
          3234,
          11621,
          24172
        ],
        "supervoxel ID 2": 1684783860,
        "supervoxel point 2": [
          3233,
          11621,
          24172
        ],
        "body point 1": [
          3234,
          11621,
          24172
        ],
        "body point 2": [
          3233,
          11621,
          24172
        ],
        "default body ID 1": 1499248377,
        "default body ID 2": 1684783860,
        "default bounding box 1": {
          "minvoxel": [0, 9088, 16960],
          "maxvoxel": [9087, 25535, 36287]
        },
        "default bounding box 2": {
          "minvoxel": [3072, 11520, 23872],
          "maxvoxel": [3263, 11647, 24191]
        }
      },
    ]
}
"""


def edges_to_assignment(df, gray_source, seg_source, sv_as_body=False, output_path=None, shuffle=False, description=""):
    if isinstance(df, str):
        df = pd.read_csv(df)

    assert isinstance(df, pd.DataFrame)

    dupes = df.duplicated(['sv_a', 'sv_b']).sum()
    if dupes:
        print(f"Dropping {dupes} duplicate tasks!")
        df = df.drop_duplicates(['sv_a', 'sv_b'])
        print(f"Writing {len(df)} tasks")

    if shuffle:
        print("Shuffling task order")
        df = df.sample(frac=1)

    df = df.copy()
    for col in ['sv_a', 'sv_b', 'xa', 'ya', 'za', 'xb', 'yb', 'zb', 'x_nearby', 'y_nearby', 'z_nearby']:
        if col in df:
            df[col] = df[col].astype(int)

    tasks = []
    for row in df.fillna(0.0).itertuples():

        body_a, body_b = row.body_a, row.body_b

        try:
            box_a = [[row.body_box_x0_a, row.body_box_y0_a, row.body_box_z0_a],
                     [row.body_box_x1_a, row.body_box_y1_a, row.body_box_z1_a]]
            box_b = [[row.body_box_x0_b, row.body_box_y0_b, row.body_box_z0_b],
                     [row.body_box_x1_b, row.body_box_y1_b, row.body_box_z1_b]]
            box_a = np.asarray(box_a)
            box_b = np.asarray(box_b)
        except AttributeError:
            box_a = np.empty((2, 3), dtype=np.float)
            box_b = np.empty((2, 3), dtype=np.float)
            box_a[:] = np.nan
            box_b[:] = np.nan

        try:
            sv_box_a = [[row.sv_box_x0_a, row.sv_box_y0_a, row.sv_box_z0_a],
                        [row.sv_box_x1_a, row.sv_box_y1_a, row.sv_box_z1_a]]
            sv_box_b = [[row.sv_box_x0_b, row.sv_box_y0_b, row.sv_box_z0_b],
                        [row.sv_box_x1_b, row.sv_box_y1_b, row.sv_box_z1_b]]
            sv_box_a = np.asarray(sv_box_a)
            sv_box_b = np.asarray(sv_box_b)
        except AttributeError:
            sv_box_a = np.empty((2, 3), dtype=np.float)
            sv_box_b = np.empty((2, 3), dtype=np.float)
            sv_box_a[:] = np.nan
            sv_box_b[:] = np.nan

        if sv_as_body:
            # If presenting the task as if the supervoxels were the body,
            # then overwrite the body items with the supervoxel info instead.
            body_a, body_b = row.sv_a, row.sv_b
            box_a, box_b = sv_box_a, sv_box_b

        edge_info = {}
        for col in df.columns:
            if 'box' not in col:
                edge_info[col] = df.loc[row.Index, col]

        task = {
            "task type": "body merge",

            "supervoxel ID 1": row.sv_a,
            "supervoxel ID 2": row.sv_b,

            "supervoxel point 1": [row.xa, row.ya, row.za],
            "supervoxel point 2": [row.xb, row.yb, row.zb],

            "body point 1": [row.xa, row.ya, row.za],
            "body point 2": [row.xb, row.yb, row.zb],

            "default body ID 1": body_a,
            "default body ID 2": body_b,

            "description": description,

            "edge_info": edge_info
        }

        # Only add the bounding box keys if the box is legit
        # (Apparently the export contains NaNs sometimes and I'm not sure why...)
        if not np.isnan(box_a).any():
            box_a = box_a.astype(int).tolist()
            task["default bounding box 1"] = {"minvoxel": box_a[0], "maxvoxel": box_a[1]}

        if not np.isnan(box_b).any():
            box_b = box_b.astype(int).tolist()
            task["default bounding box 2"] = {"minvoxel": box_b[0], "maxvoxel": box_b[1]}

        tasks.append(task)

    assignment = {
        "file type": "Neu3 task list",
        "file version": 1,
        "grayscale source": gray_source,
        "segmentation source": seg_source,
        "task list": tasks
    }

    if description:
        assignment["task set description"] = description

    assignment = convert_nans(assignment)
    if output_path:
        dump_json(assignment, output_path, unsplit_int_lists=True)

    return assignment


def edges_to_assignments(df, gray_source, seg_source, sv_as_body=False, batch_size=100, output_path=None, *, shuffle=False, description=""):
    if isinstance(df, str):
        df = pd.read_csv(df)
    assert isinstance(df, pd.DataFrame)

    dupes = df.duplicated(['sv_a', 'sv_b']).sum()
    if dupes:
        print(f"Dropping {dupes} duplicate tasks!")
        df = df.drop_duplicates(['sv_a', 'sv_b'])
        print(f"Writing {len(df)} tasks")

    if shuffle:
        print("Shuffling task order")
        df = df.sample(frac=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    assignments = []
    for i, batch_df in enumerate(iter_batches(df, batch_size)):
        if output_path:
            base, _ = os.path.splitext(output_path)
            batch_path = f"{base}-{i:03d}.json"
        else:
            batch_path = None

        a = edges_to_assignment(batch_df, gray_source, seg_source, sv_as_body, batch_path, description=description)
        assignments.append(a)


if __name__ == "__main__":
    VNC_GRAY = "precomputed://gs://flyem-vnc-2-26-213dba213ef26e094c16c860ae7f4be0/v3_emdata_clahe_xy/jpeg"
    VNC_BASE = "precomputed://gs://vnc-v3-seg-3d2f1c08fd4720848061f77362dc6c17/rc4_wsexp"
    VNC_AGGLO = "precomputed://gs://vnc-v3-seg-3d2f1c08fd4720848061f77362dc6c17/rc4_wsexp_rsg32_16_sep_8_sep1e6"

    # bq_export_path = '/Users/bergs/Downloads/rsg32-task-table.csv'
    # agglo_output_path = '/tmp/rsg32-tasks-agglo-ids.json'
    # sv_output_path = '/tmp/rsg32-tasks-sv-ids.json'

    # bq_export_path = '/Users/bergs/Downloads/rsg16-task-table.csv'
    # agglo_output_path = '/tmp/rsg16-tasks-agglo-ids.json'
    # sv_output_path = '/tmp/rsg16-tasks-sv-ids.json'

    # _ = edges_to_assignment(bq_export_path, VNC_GRAY, VNC_AGGLO, output_path=agglo_output_path)
    # _ = edges_to_assignment(bq_export_path, VNC_GRAY, VNC_BASE, sv_as_body=True, output_path=sv_output_path)

    os.chdir('/tmp')

    # from google.cloud import bigquery
    # client = bigquery.Client('janelia-flyem')

    # def fetch_focused_task_table(rsg):
    #     q = f"""\
    #         select
    #         floor(score/0.1)/10 as scorebin,
    #         floor(least(sv_tbars_a, sv_tbars_b) / 10)*10 tbarbin,
    #         *  
    #         from vnc_rc4_focused_exports.all_rsg{rsg}_2tbar_neuropil_edges
    #         where
    #         least(score_ab, score_ba) > 0.1
    #         and sv_tbars_a >= 2 and sv_tbars_b >= 2
    #         order by tbarbin desc, score desc
    #     """
    #     r = client.query(q).result()
    #     df = pd.DataFrame((row.values() for row in tqdm_proxy(r, total=r.total_rows)), columns=[f.name for f in r.schema])
    #     return df
    
    # tasks_8 = fetch_focused_task_table(8)
    # tasks_16 = fetch_focused_task_table(16)
    # tasks_32 = fetch_focused_task_table(32)

    # tasks_8.to_csv('focused-edges/focused-rsg8.csv', header=True, index=False)
    # tasks_16.to_csv('focused-edges/focused-rsg16.csv', header=True, index=False)
    # tasks_32.to_csv('focused-edges/focused-rsg32.csv', header=True, index=False)

    # dfs = []
    # for rsg in [8, 16, 32]:
    #     df = pd.read_csv(f'/tmp/focused-edges/focused-rsg{rsg}.csv')
    #     df['rsg'] = rsg
    #     dfs.append(df)
    # df = pd.concat(dfs, ignore_index=True)
    # sv_output_path = '/tmp/vnc-sv-focused-assignments/focused.json'
    # _ = edges_to_assignments(df, VNC_GRAY, VNC_BASE, sv_as_body=True, output_path=sv_output_path)

    # np.random.seed(0)
    # p = '/tmp/excluded-1M-1M-edges-from-rsg8.csv'
    # description = "excluded-1M-1M-from-rsg8-sv-ids"
    # _ = edges_to_assignments(p, VNC_GRAY, VNC_BASE, sv_as_body=True,
    #                          output_path=f'/tmp/{description}/tasks.json',
    #                          shuffle=True,
    #                          description=description)

    # np.random.seed(0)
    # p = '/tmp/excluded-1M-1M-edges-from-rsg8.csv'
    # description = "excluded-1M-1M-from-rsg8-AGGLO-ids"
    # _ = edges_to_assignments(p, VNC_GRAY, VNC_AGGLO, sv_as_body=False,
    #                          output_path=f'/tmp/{description}/tasks.json',
    #                          shuffle=True,
    #                          description=description)

    # np.random.seed(0)
    # p = '/Users/bergs/workspace/vnc-focused-queries/tables/16nm-would-be-violations.csv'
    # description = "would-be-10M-10M-violations-from-rsg16"
    # _ = edges_to_assignments(p, VNC_GRAY, VNC_BASE, sv_as_body=True,
    #                          output_path=f'/tmp/{description}/tasks.json',
    #                          shuffle=True,
    #                          description=description)

    # np.random.seed(0)
    # p = '/Users/bergs/workspace/vnc-focused-queries/tables/synth-big-sv-direct.csv'
    # description = "synth32_max1000-big-sv-direct"
    # _ = edges_to_assignments(p, VNC_GRAY, VNC_BASE, sv_as_body=True,
    #                          output_path=f'/tmp/{description}/tasks.json',
    #                          shuffle=True,
    #                          description=description)

    # np.random.seed(0)
    # p = '/Users/bergs/workspace/vnc-focused-queries/tables/synth-big-body-direct.csv'
    # description = "synth32_max1000-big-body-direct"
    # _ = edges_to_assignments(p, VNC_GRAY, VNC_BASE, sv_as_body=True,
    #                          output_path=f'/tmp/{description}/tasks.json',
    #                          shuffle=True,
    #                          description=description)

    # np.random.seed(0)
    # p = '/Users/bergs/workspace/vnc-focused-queries/tables/synth-score-5-10-direct.csv'
    # description = "synth32_max1000-score-5-10-direct"
    # _ = edges_to_assignments(p, VNC_GRAY, VNC_BASE, sv_as_body=True,
    #                          output_path=f'/tmp/{description}/tasks.json',
    #                          shuffle=True,
    #                          description=description)

    # np.random.seed(0)
    # p = '/Users/bergs/workspace/vnc-focused-queries/tables/synth-1-to-2-tbar-direct.csv'
    # description = "synth-1-to-2-tbar-direct"
    # _ = edges_to_assignments(p, VNC_GRAY, VNC_BASE, sv_as_body=True,
    #                          output_path=f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}/tasks.json',
    #                          shuffle=True,
    #                          description=description)

    # np.random.seed(0)
    # p = '/Users/bergs/workspace/vnc-focused-queries/tables/remaining-rsg-favorite-2tbar-direct.csv'
    # description = "remaining-rsg-favorite-2tbar-direct"
    # _ = edges_to_assignments(p, VNC_GRAY, VNC_BASE, sv_as_body=True,
    #                          output_path=f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}/tasks.json',
    #                          shuffle=True,
    #                          description=description)

    # np.random.seed(0)
    # p = '/Users/bergs/workspace/vnc-focused-queries/tables/remaining-rsg-favorite-tiny-boi-direct.csv'
    # description = "remaining-rsg-favorite-tiny-boi-direct"
    # _ = edges_to_assignments(p, VNC_GRAY, VNC_BASE, sv_as_body=True,
    #                          output_path=f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}/tasks.json',
    #                          shuffle=True,
    #                          description=description)

    # np.random.seed(0)
    # p = '/Users/bergs/workspace/vnc-focused-queries/tables/lowscore-0.03-rsg-favorite-boi-direct.csv'
    # description = "lowscore-0.03-rsg-favorite-boi-direct"
    # _ = edges_to_assignments(p, VNC_GRAY, VNC_BASE, sv_as_body=True,
    #                          output_path=f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}/tasks.json',
    #                          shuffle=True,
    #                          description=description)

    # np.random.seed(0)
    # p = '/Users/bergs/workspace/vnc-focused-queries/tables/lowscore-0.02-rsg-favorite-boi-direct.csv'
    # description = "lowscore-0.02-rsg-favorite-boi-direct"
    # _ = edges_to_assignments(p, VNC_GRAY, VNC_BASE, sv_as_body=True,
    #                          output_path=f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}/tasks.json',
    #                          shuffle=True,
    #                          description=description)

    # np.random.seed(0)
    # p = '/Users/bergs/workspace/vnc-focused-queries/tables/lowscore-0.01-rsg-favorite-boi-direct.csv'
    # description = "lowscore-0.01-rsg-favorite-boi-direct"
    # _ = edges_to_assignments(p, VNC_GRAY, VNC_BASE, sv_as_body=True,
    #                          output_path=f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}/tasks.json',
    #                          shuffle=True,
    #                          description=description)

    # np.random.seed(0)
    # p = '/Users/bergs/workspace/vnc-focused-queries/tables/lowscore-0.005-rsg-favorite-boi-direct.csv'
    # description = "lowscore-0.005-rsg-favorite-boi-direct"
    # _ = edges_to_assignments(p, VNC_GRAY, VNC_BASE, sv_as_body=True,
    #                          output_path=f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}/tasks.json',
    #                          shuffle=True,
    #                          description=description)

    # for score in np.arange(0.001, 0.005, 0.001):
    #     print(f"Processing tasks for score {score:.3f}")
    #     np.random.seed(0)
    #     p = f'/Users/bergs/workspace/vnc-focused-queries/tables/lowscore-{score:.3f}-rsg-favorite-boi-direct.csv'
    #     description = f"lowscore-{score:.3f}-rsg-favorite-boi-direct"
    #     _ = edges_to_assignments(p, VNC_GRAY, VNC_BASE, sv_as_body=True,
    #                              output_path=f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}/tasks.json',
    #                              shuffle=True,
    #                              description=description)

    # np.random.seed(0)
    # description = "lowscore-0.0005-rsg-favorite-boi-direct"
    # p = f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}.csv'
    # _ = edges_to_assignments(p, VNC_GRAY, VNC_BASE, sv_as_body=True,
    #                          output_path=f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}/tasks.json',
    #                          shuffle=True,
    #                          description=description)

    # np.random.seed(0)
    # description = "lowscore-0.0001-rsg-favorite-boi-direct"
    # p = f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}.csv'
    # _ = edges_to_assignments(p, VNC_GRAY, VNC_BASE, sv_as_body=True,
    #                          output_path=f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}/tasks.json',
    #                          shuffle=True,
    #                          description=description)

    # np.random.seed(0)
    # description = "focused-2021-05-02"
    # p = f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}.csv'
    # VNC_DVID_SRC = 'dvid://https://emdata5-avempartha.janelia.org/d9670ddd1681495db4c10865bf4819e4/segmentation'
    # _ = edges_to_assignments(p, VNC_GRAY, VNC_DVID_SRC, sv_as_body=True,
    #                          output_path=f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}/tasks.json',
    #                          shuffle=True,
    #                          description=description)

    # np.random.seed(0)
    # description = "unapplied-merges-2021-05-03"
    # p = f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}.csv'
    # VNC_DVID_SRC = 'dvid://https://emdata5-avempartha.janelia.org/d9670ddd1681495db4c10865bf4819e4/segmentation'
    # _ = edges_to_assignments(p, VNC_GRAY, VNC_DVID_SRC, sv_as_body=True,
    #                          output_path=f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}/tasks.json',
    #                          shuffle=True,
    #                          description=description)

    np.random.seed(0)
    description = "unapplied-merges-2021-05-03"
    p = f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}.csv'
    VNC_DVID_SRC = 'dvid://https://emdata5-avempartha.janelia.org/d9670ddd1681495db4c10865bf4819e4/segmentation'
    _ = edges_to_assignments(p, VNC_GRAY, VNC_DVID_SRC, sv_as_body=True,
                             output_path=f'/Users/bergs/workspace/vnc-focused-queries/tables/{description}/tasks.json',
                             shuffle=True,
                             description=description)

