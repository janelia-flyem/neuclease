import json
import numpy as np
import pandas as pd

ASSIGNMENT_EXAMPLE = """\
{
  "file type": "Neu3 task list",
  "file version": 1,
  "grayscale source": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg",
  "segmentation source": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.1/segmentation",
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


def edges_to_assignment(path, gray_source, seg_source, sv_as_body=False, output_path=None):
    df = pd.read_csv(path)

    dupes = df.duplicated(['sv_a', 'sv_b']).sum()
    if dupes:
        print(f"Dropping {dupes} duplicate tasks!")
        df = df.drop_duplicates(['sv_a', 'sv_b'])
        print(f"Writing {len(df)} tasks")

    tasks = []
    for row in df.itertuples():

        body_a, body_b = row.body_a, row.body_b
        box_a = [[row.body_box_x0_a, row.body_box_y0_a, row.body_box_z0_a],
                 [row.body_box_x1_a, row.body_box_y1_a, row.body_box_z1_a]]
        box_b = [[row.body_box_x0_b, row.body_box_y0_b, row.body_box_z0_b],
                 [row.body_box_x1_b, row.body_box_y1_b, row.body_box_z1_b]]
        box_a = np.asarray(box_a)
        box_b = np.asarray(box_b)

        sv_box_a = [[row.sv_box_x0_a, row.sv_box_y0_a, row.sv_box_z0_a],
                    [row.sv_box_x1_a, row.sv_box_y1_a, row.sv_box_z1_a]]
        sv_box_b = [[row.sv_box_x0_b, row.sv_box_y0_b, row.sv_box_z0_b],
                    [row.sv_box_x1_b, row.sv_box_y1_b, row.sv_box_z1_b]]

        sv_box_a = np.asarray(sv_box_a)
        sv_box_b = np.asarray(sv_box_b)

        if sv_as_body:
            # If presenting the task as if the supervoxels were the body,
            # then overwrite the body items with the supervoxel info instead.
            body_a, body_b = row.sv_a, row.sv_b
            box_a, box_b = sv_box_a, sv_box_b

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
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(assignment, f, indent=2)

    return assignment


if __name__ == "__main__":
    VNC_GRAY = "precomputed://gs://flyem-vnc-2-26-213dba213ef26e094c16c860ae7f4be0/v3_emdata_clahe_xy/jpeg"
    VNC_BASE = "precomputed://gs://vnc-v3-seg-3d2f1c08fd4720848061f77362dc6c17/rc4_wsexp"
    VNC_AGGLO = "precomputed://gs://vnc-v3-seg-3d2f1c08fd4720848061f77362dc6c17/rc4_wsexp_rsg32_16_sep_8_sep1e6"

    # bq_export_path = '/Users/bergs/Downloads/rsg32-task-table.csv'
    # agglo_output_path = '/tmp/rsg32-tasks-agglo-ids.json'
    # sv_output_path = '/tmp/rsg32-tasks-sv-ids.json'

    bq_export_path = '/Users/bergs/Downloads/rsg16-task-table.csv'
    agglo_output_path = '/tmp/rsg16-tasks-agglo-ids.json'
    sv_output_path = '/tmp/rsg16-tasks-sv-ids.json'

    _ = edges_to_assignment(bq_export_path, VNC_GRAY, VNC_AGGLO, output_path=agglo_output_path)
    _ = edges_to_assignment(bq_export_path, VNC_GRAY, VNC_BASE, sv_as_body=True, output_path=sv_output_path)
