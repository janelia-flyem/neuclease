# @license
# Copyright 2017 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
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

def load_edges2(path):
  edges = []
  df = pandas.read_csv(path, sep=',', dtype={'segment_a': np.uint64, 'segment_b': np.uint64, 'score': np.float64, 'x': np.int64, 'y': np.int64, 'z': np.int64})
  return df


def write_db(edges_csv_path, output_path):

  print('Loading edges')
  edges = load_edges2(edges_csv_path)

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
  ap.add_argument('csv', help='Path to CSV file specifying edges.')
  ap.add_argument('db', help='Output path to sqlite3 db.')
  args = ap.parse_args()

  write_db(args.csv, args.db)
