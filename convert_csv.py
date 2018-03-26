# @license
# Copyright 2017 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Author: Jeremy Maitin-Shepard (https://github.com/jbms)
"""

import pandas
import sqlite3
import neuroglancer.equivalence_map
import argparse
import numpy as np


def load_edges2(path, col_format):
    print('Loading edges')
    
    if path.endswith('.npy'):
        df_raw = pandas.DataFrame(np.load(path))
    elif path.endswith('.csv'):
        if col_format == 'jeremy':
            df_raw = pandas.read_csv(path, sep=',', dtype={ 'segment_a': np.uint64,
                                                        'segment_b': np.uint64,
                                                        'score': np.float64,
                                                        'x': np.int64,
                                                        'y': np.int64,
                                                        'z': np.int64})
        elif col_format == 'michal':
            # Convert Michal's FFN export CSV to the same columns as Jeremy's celis exports.
            df_raw = pandas.read_csv(path, dtype={'id_a': np.uint64,
                                                  'id_b': np.uint64,
                                                  'score': np.float64,
                                                  'xa': np.int64,
                                                  'ya': np.int64,
                                                  'za': np.int64,
                                                  'xb': np.int64,
                                                  'yb': np.int64,
                                                  'zb': np.int64})
        else:
            raise RuntimeError("Unknown format: {}".format(col_format))

    
    if col_format == 'jeremy':
        return df_raw
    elif col_format == 'michal':
        df = pandas.DataFrame(columns=['segment_a', 'segment_b', 'score', 'x', 'y', 'z'])
        
        df['segment_a'] = df_raw['id_a']
        df['segment_b'] = df_raw['id_b']
        df['score'] = df_raw['score']
        df['x'] = (df_raw['xa'] + df_raw['xb']) // 2
        df['y'] = (df_raw['ya'] + df_raw['yb']) // 2
        df['z'] = (df_raw['za'] + df_raw['zb']) // 2
    else:
        raise RuntimeError("Unknown format: {}".format(col_format))

    df['segment_a'] = df['segment_a'].astype(np.uint64)
    df['segment_b'] = df['segment_b'].astype(np.uint64)
    df['score'] = df['score'].astype(np.float64)
    df['x'] = df['x'].astype(np.int64)
    df['y'] = df['y'].astype(np.int64)
    df['z'] = df['z'].astype(np.int64)

    # Remove all self-edges (self loops)
    df = df.query('segment_a != segment_b')

    return df


def write_db(edges, output_path):

    all_eqs = neuroglancer.equivalence_map.EquivalenceMap()
    print('Creating equivalence map for agglomeration')
    for a, b in edges[['segment_a', 'segment_b']].values:
        all_eqs.union(a, b)

    conn = sqlite3.connect(output_path)
    c = conn.cursor()

    c.execute('CREATE TABLE supervoxels (supervoxel_id INTEGER, agglo_id INTEGER)')
    c.execute('CREATE INDEX supervoxels_by_supervoxel_id_index ON supervoxels (supervoxel_id)')
    c.execute('CREATE INDEX supervoxels_by_agglo_id_index ON supervoxels (agglo_id)')
    c.execute('CREATE TABLE edges (agglo_id INTEGER, segment_a INTEGER, segment_b INTEGER, score REAL, x INTEGER, y INTEGER, z INTEGER)')
    c.execute('CREATE INDEX edges_by_agglo_id_index ON edges (agglo_id)')

    print('Writing supervoxels table')
    c.executemany('INSERT INTO supervoxels VALUES (?,?)',
                                ((int(x), int(all_eqs[x])) for x in all_eqs.keys()))

    print('Writing edges table')
    c.executemany(
            'INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?)',
            ((int(all_eqs[segment_a]), int(segment_a), int(segment_b), float(score), int(x), int(y), int(z))
             for (segment_a, segment_b), score,
             (x, y, z) in zip(edges[['segment_a', 'segment_b']].values, edges['score']
                                                .values, edges[['x', 'y', 'z']].values)))
    print('Committing')
    conn.commit()
    conn.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--format', choices=['jeremy', 'michal'], default='celis')
    ap.add_argument('csv', help='Path to CSV file specifying edges.')
    ap.add_argument('db', help='Output path to sqlite3 db.')
    args = ap.parse_args()

    edges_df = load_edges2(args.csv, args.format)
    write_db(edges_df, args.db)
