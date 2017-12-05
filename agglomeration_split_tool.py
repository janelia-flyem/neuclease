import sys
import argparse
import collections
import copy
import heapq
import json
import re
import sqlite3
import logging

import numpy as np
import six

import neuroglancer

debug_graph = False
verbose_merging = False


def normalize_edge(id_pair):
  (id_a, id_b) = id_pair
  if id_a > id_b:
    id_a, id_b = id_b, id_a
  return id_a, id_b


class GreedyMulticut(object):

  def __init__(self, combine_edges, edge_priority):
    # Contains (score, edge_map_value) tuple values in heap order.  The
    # edge_map_value is the actual corresponding value in edge_map, not a copy.
    self.edge_heap = []

    # Maps segment_id -> set of segment_id neighbors.
    self.regions = dict()

    # Maps (id_a, id_b) -> edge_map_value=[score, key, edge_object]
    self.edge_map = dict()
    self.combine_edges = combine_edges
    self.edge_priority = edge_priority
    self.num_valid_edges = 0
    self._initialized = False

  def add_edge(self, id_pair, edge):
    (id_a, id_b) = id_pair
    id_a, id_b = normalize_edge((id_a, id_b))
    self.regions.setdefault(id_a, set()).add(id_b)
    self.regions.setdefault(id_b, set()).add(id_a)
    key = (id_a, id_b)
    entry = self.edge_map.get(key, None)
    if entry is not None:
      edge_data = entry[2] = self.combine_edges(entry[0], edge)
      entry[0] = self.edge_priority(edge_data)
    else:
      entry = self.edge_map[key] = [self.edge_priority(edge), key, edge]
      self.num_valid_edges += 1
    if self._initialized:
      self._add_to_heap(entry)

  def _initialize_heap(self):
    if self._initialized:
      return
    for key in self.edge_map:
      entry = self.edge_map[key]
      self._add_to_heap(entry)
    self._initialized = True

  def _add_to_heap(self, entry):
    heapq.heappush(self.edge_heap, (entry[0], entry))

  def remove_edge_from_heap(self, segment_ids):
    """Remove an edge from the heap."""
    self._initialize_heap()
    key = normalize_edge(segment_ids)
    if key in self.edge_map:
      self.edge_map[key][0] = None
      self.num_valid_edges -= 1

  def check_consistency(self):
    self._initialize_heap()
    expected_regions = dict()
    for key, entry in six.viewitems(self.edge_map):
      assert entry[1] == key
      expected_regions.setdefault(key[0], set()).add(key[1])
      expected_regions.setdefault(key[1], set()).add(key[0])

    assert expected_regions == self.regions

    num_valid_edges = 0
    for e in self.edge_heap:
      if self._is_valid_heap_entry(e):
        num_valid_edges += 1
    assert num_valid_edges == self.num_valid_edges

  def merge(self, id_pair):
    (id_a, id_b) = id_pair
    self._initialize_heap()
    id_a, id_b = normalize_edge((id_a, id_b))
    if (id_a, id_b) not in self.edge_map:
      raise KeyError
    for neighbor in self.regions[id_b]:
      if neighbor == id_a:
        continue
      expired_ids = normalize_edge((neighbor, id_b))
      new_ids = normalize_edge((neighbor, id_a))
      new_edge = self.edge_map.get(new_ids)
      expired_edge = self.edge_map[expired_ids]
      if new_edge is not None:
        edge_data = new_edge[2] = self.combine_edges(new_edge[2],
                                                     expired_edge[2])
        if new_edge[0] is not None:
          self.num_valid_edges -= 1
        if expired_edge[0] is not None:
          self.num_valid_edges -= 1
        self.num_valid_edges += 1
        new_edge[0] = self.edge_priority(edge_data)
        self._add_to_heap(new_edge)
      else:
        self.regions[neighbor].add(id_a)
        self.regions[id_a].add(neighbor)
        self.edge_map[new_ids] = expired_edge
        expired_edge[1] = new_ids
        # No need to add to heap, since score hasn't changed.
      del self.edge_map[expired_ids]
      self.regions[neighbor].remove(id_b)
    del self.regions[id_b]
    self.regions[id_a].remove(id_b)
    del self.edge_map[(id_a, id_b)]
    self.num_valid_edges -= 1

  def _is_valid_heap_entry(self, heap_entry):
    score, entry = heap_entry
    expected_entry = self.edge_map.get(entry[1])
    if entry is not expected_entry or entry[0] is not score:
      return None
    else:
      return entry

  def get_next_edge(self):
    self._initialize_heap()
    while True:
      if self.num_valid_edges == 0:
        return None
      heap_entry = self.edge_heap[0]
      entry = self._is_valid_heap_entry(heap_entry)
      if entry is None:
        heapq.heappop(self.edge_heap)
      else:
        return entry


Edge = collections.namedtuple('Edge', ['segment_ids', 'score', 'position'])


def load_edges(path):
  edges = []
  with open(path, 'r') as f:
    f.readline()
    for line in f:
      parts = line.split(',')
      segment_a = int(parts[0].strip())
      segment_b = int(parts[1].strip())
      score = float(parts[2].strip())
      position = (int(parts[3].strip()), int(parts[4].strip()),
                  int(parts[5].strip()))
      edges.append(
          Edge(segment_ids=(segment_a, segment_b), score=score,
               position=position))
  return edges


def load_split_seeds(path):
  with open(path, 'r') as f:
    raw_seeds = json.loads(f.read())
  seeds = collections.OrderedDict()
  for component in raw_seeds:
    seeds.setdefault(component['label'], []).extend(component['supervoxels'])
  return seeds


def build_graph(edges):
  logging.info('Building graph with %d edges', len(edges))

  def combine_edges(a, b):
    return a + b

  def edge_priority(x):
    return x

  greedy_multicut = GreedyMulticut(
      combine_edges=combine_edges,
      edge_priority=edge_priority,
  )
  for edge in edges:
    greedy_multicut.add_edge(edge.segment_ids, edge.score)
  return greedy_multicut


class AgglomerationGraph(object):

  def __init__(self, conn):
    self.conn = conn

  def get_agglo_id(self, supervoxel_id):
    c = self.conn.cursor()
    c.execute('SELECT agglo_id FROM supervoxels WHERE supervoxel_id=?',
              (int(supervoxel_id),))
    result = c.fetchone()
    if result is None:
      return supervoxel_id
    else:
      return result[0]

  def get_agglo_members(self, agglo_id):
    c = self.conn.cursor()
    c.execute('SELECT supervoxel_id FROM supervoxels WHERE agglo_id=?',
              (int(agglo_id),))
    return [row[0] for row in c.fetchall()]

  def get_agglo_edges(self, agglo_id):
    c = self.conn.cursor()
    c.execute(
        'SELECT segment_a, segment_b, score, x, y, z FROM edges WHERE agglo_id=?',
        (int(agglo_id),))
    return [
        Edge(segment_ids=(row[0], row[1]), score=row[2],
             position=(row[3], row[4], row[5])) for row in c.fetchall()
    ]

def _make_supervoxel_map(graph, split_seeds):
  supervoxel_map = dict()
  agglo_ids = dict()

  for label, component in six.viewitems(split_seeds):
    for seed in component:
      supervoxel_id = seed['supervoxel_id']
      agglo_id = graph.get_agglo_id(supervoxel_id)
      if agglo_id == 0:
        continue
      agglo_ids.setdefault(agglo_id, []).append((label, seed))
      supervoxel_map.setdefault(supervoxel_id, set()).add(label)
  return agglo_ids, supervoxel_map

def do_split(graph, split_seeds, agglo_id=None):

  agglo_ids, supervoxel_map = _make_supervoxel_map(graph, split_seeds)

  if agglo_id is None:

    agglo_id_counts = {
        agglo_id: sum(z[1]['count'] for z in seeds)
        for agglo_id, seeds in six.viewitems(agglo_ids)
    }

    agglo_id = max(agglo_ids, key=lambda x: agglo_id_counts[x])

    if len(agglo_ids) > 1:
      logging.info('Warning: more than one agglomerated component.  ' +
                   'Choosing component %d with maximum number of seed points.',
                   agglo_id)
      logging.info('agglo_id_counts = %r', agglo_id_counts)

  graph = build_graph(graph.get_agglo_edges(agglo_id))
  if debug_graph:
    graph.check_consistency()

  cur_eqs = neuroglancer.EquivalenceMap()
  logging.info('Agglomerating')
  threshold = float('inf')
  while True:
    entry = graph.get_next_edge()
    if entry is None:
      if verbose_merging:
        logging.info('Stopping because entry is None')
      break
    if entry[0] > threshold:
      if verbose_merging:
        logging.info('Stopping because edge score %r is > threshold %r',
                     entry[0], threshold)
      break
    segment_ids = entry[1]
    seeds_a = supervoxel_map.get(segment_ids[0])
    seeds_b = supervoxel_map.get(segment_ids[1])
    if ((seeds_a is not None and len(seeds_a) > 1) or
        (seeds_b is not None and len(seeds_b) > 1) or
        (seeds_a is not None and seeds_b is not None and seeds_a != seeds_b)):
      if verbose_merging:
        logging.info('Excluding edge %r because of seeds: %r %r', segment_ids,
                     seeds_a, seeds_b)
      graph.remove_edge_from_heap(segment_ids)
      continue
    if verbose_merging:
      logging.info('Merging %r with score %r', segment_ids, entry[0])
    graph.merge(segment_ids)
    if debug_graph:
      graph.check_consistency()

    new_id = cur_eqs.union(*segment_ids)
    new_seeds = seeds_a or seeds_b
    if new_seeds:
      supervoxel_map[new_id] = new_seeds

  return dict(agglo_id=agglo_id, cur_eqs=cur_eqs, supervoxel_map=supervoxel_map)


def display_split_result(graph, agglo_id, cur_eqs, supervoxel_map, split_seeds,
                         image_url, segmentation_url):

  agglo_members = set(graph.get_agglo_members(agglo_id))
  state = neuroglancer.ViewerState()
  state.layers.append(name='image', layer=neuroglancer.ImageLayer(
      source=image_url))
  state.layers.append(
      name='original',
      layer=neuroglancer.SegmentationLayer(
          source=segmentation_url,
          segments=agglo_members,
      ),
      visible=False,
  )
  state.layers.append(
      name='isolated-supervoxels',
      layer=neuroglancer.SegmentationLayer(
          source=segmentation_url,
          segments=set(x for x, seeds in six.viewitems(supervoxel_map)
                       if len(seeds) > 1),
      ),
      visible=False,
  )
  state.layers.append(name='split', layer=neuroglancer.SegmentationLayer(
      source=segmentation_url,
      equivalences=cur_eqs,
      segments=set(cur_eqs[x] for x in agglo_members),
  ))
  for label, component in six.viewitems(split_seeds):
    state.layers.append(
        name='seed%d' % label,
        layer=neuroglancer.PointAnnotationLayer(
            points=[seed['position'] for seed in component],
        ),
    )

  state.show_slices = False
  state.layout = '3d'
  all_seed_points = [
      seed['position']
      for component in six.viewvalues(split_seeds) for seed in component
  ]
  state.voxel_coordinates = np.mean(all_seed_points, axis=0)
  state.perspective_zoom = 140
  return state


def _get_layer_seed_label(name):
  m = re.match('^seed([0-9]+)$', name)
  if m is not None:
    try:
      return int(m.group(1))
    except ValueError:
      pass
  return None


def _set_viewer_seeds(s, seeds):
  for layer in list(s.layers):
    label = _get_layer_seed_label(layer.name)
    if label is not None and label not in seeds:
      del s.layers[label]
  for label, component in six.viewitems(seeds):
    s.layers['seed%d' % label] = neuroglancer.PointAnnotationLayer(
        points=[x['position'] for x in component],
    )


class InteractiveSplitter(object):

  def __init__(self, graph, agglo_id, image_url, segmentation_url,
               split_seeds=None):
    self.graph = graph
    self.agglo_id = agglo_id
    self.image_url = image_url
    self.segmentation_url = segmentation_url
    self.seed_position_to_supervoxel_id_map = dict()
    self.last_seeds = collections.OrderedDict()
    self.last_split_result = None
    if split_seeds is not None:
      split_seeds = collections.OrderedDict(
          (int(label),
           [{
               'count': int(seed['count']),
               'position': tuple(int(x) for x in seed['position']),
               'supervoxel_id': int(seed['supervoxel_id'])
           } for seed in supervoxels])
          for label, supervoxels in six.viewitems(split_seeds))
      self._update_split_result(split_seeds)
      self._update_seed_position_map(self.last_seeds)
    if agglo_id is None:
      agglo_id = self.last_split_result['agglo_id']
    self.agglo_members = set(self.graph.get_agglo_members(agglo_id))

    viewer = self.viewer = neuroglancer.Viewer()

    seed_keys = ['keyd', 'keyf', 'keyg', 'keyh', 'keyj']
    max_seeds = len(seed_keys)

    for label in range(1, max_seeds + 1):

      def add_seed(s, label=label):
        self._add_seed(s, label)

      viewer.actions.add('mark-seed-%d' % label, add_seed)
    with viewer.config_state.txn() as s:
      s.status_messages['help'] = (
          'MARKING SEEDS: ' + ' | '.join('%s=mark-seed-%d' % (key, i + 1)
                                           for i, key in enumerate(seed_keys)))
      for i, key in enumerate(seed_keys):
        s.input_event_bindings.viewer[key] = 'mark-seed-%d' % (i + 1)

    with viewer.txn() as s:
      s.perspective_zoom = 140
      s.layers.append(
          name='image',
          layer=neuroglancer.ImageLayer(source=self.image_url),
      )
      s.layers.append(
          name='original',
          layer=neuroglancer.SegmentationLayer(
              source=self.segmentation_url,
              segments=self.agglo_members,
          ),
      )
      s.layers.append(
          name='split-result',
          layer=neuroglancer.SegmentationLayer(
              source=self.segmentation_url,
              segments=self.agglo_members,
          ),
      )
      s.layers.append(
          name='isolated-supervoxels',
          layer=neuroglancer.SegmentationLayer(
              source=segmentation_url,
          ),
          visible=False,
      )
      _set_viewer_seeds(s, self.last_seeds)
      self._update_state(s)
    viewer.shared_state.add_changed_callback(
        lambda: viewer.defer_callback(self._handle_state_changed))

  def _handle_state_changed(self):
    seeds = self._get_seeds(self.viewer.state)
    if seeds == self.last_seeds:
      return
    with self.viewer.txn() as s:
      self._update_state(s)

  def _update_split_result(self, seeds):
    if seeds != self.last_seeds or self.last_split_result is None:
      self.last_seeds = seeds
      self.last_split_result = do_split(
          graph=self.graph, split_seeds=self.last_seeds, agglo_id=self.agglo_id)
      return True
    return False

  def _update_seed_position_map(self, seeds):
    m = self.seed_position_to_supervoxel_id_map
    m.clear()
    for label, component in six.viewitems(seeds):
      for seed in component:
        m[(label, seed['position'])] = seed['supervoxel_id']

  def _add_seed(self, s, label):
    seeds = self._get_seeds(s.viewer_state)
    supervoxel_id = s.selected_values['original']
    if supervoxel_id is None:
      m = s.selected_values['split-result']
      if m is not None:
        if isinstance(m, neuroglancer.MapEntry):
          supervoxel_id = m.key
        else:
          supervoxel_id = m
    mouse_voxel_coordinates = s.mouse_voxel_coordinates
    if mouse_voxel_coordinates is None or supervoxel_id is None or supervoxel_id == 0:
      return
    seeds.setdefault(label, []).append(
        dict(
            supervoxel_id=supervoxel_id,
            position=tuple(int(x) for x in mouse_voxel_coordinates),
            count=1,
        ),
    )
    self._update_seed_position_map(seeds)
    new_state = copy.deepcopy(s.viewer_state)
    _set_viewer_seeds(new_state, seeds)
    self._update_state(new_state)
    self.viewer.set_state(new_state)

  def _update_state(self, s):
    seeds = self._get_seeds(s)
    _set_viewer_seeds(s, seeds)
    self._update_split_result(seeds)
    split_result = self.last_split_result
    self._show_split_result(
        s,
        cur_eqs=split_result['cur_eqs'],
        supervoxel_map=split_result['supervoxel_map'],
    )
    seed_layers = ['seed%d' % label for label in self.last_seeds]
    s.layout = neuroglancer.row_layout([
        neuroglancer.LayerGroupViewer(
            layout='3d',
            layers=['image', 'original', 'isolated-supervoxels'] + seed_layers),
        neuroglancer.LayerGroupViewer(
            layout='3d', layers=['image', 'split-result'] + seed_layers),
    ])

  def _show_split_result(self, s, cur_eqs, supervoxel_map):
    s.layers['isolated-supervoxels'].segments = set(
        x for x, seeds in six.viewitems(supervoxel_map) if len(seeds) > 1)
    split_layer = s.layers['split-result']
    split_layer.equivalences = cur_eqs
    split_layer.segments = set(cur_eqs[x] for x in self.agglo_members)

  def _get_seeds(self, s):
    seeds = collections.OrderedDict()
    for layer in s.layers:
      label = _get_layer_seed_label(layer.name)
      if label is None:
        continue
      component = []
      for point in layer.points:
        point = tuple(int(x) for x in np.cast[np.int64](point))
        supervoxel_id = self.seed_position_to_supervoxel_id_map.get((label,
                                                                     point))
        if supervoxel_id is None:
          continue
        component.append(
            dict(supervoxel_id=supervoxel_id, position=point, count=1))
      seeds[label] = component
    return seeds


def run_batch(args, graph):
  for path in args.split_seeds:
    split_seeds = load_split_seeds(path)
    split_result = do_split(graph=graph, split_seeds=split_seeds,
                            agglo_id=args.agglo_id)
    state = display_split_result(graph=graph, split_seeds=split_seeds,
                                 image_url=args.image_url,
                                 segmentation_url=args.segmentation_url, **split_result)
    print('<p><a href="%s">%s</a></p>' % (neuroglancer.to_url(state), path))


def run_interactive(args, graph, agglo_id=None, split_seeds=None):
  # Make splitter a global variable so that it is accessible from the
  # interactive `python -i` shell.
  global splitter

  neuroglancer.set_static_content_source(
      url='https://neuroglancer-demo.appspot.com/python')
  neuroglancer.set_server_bind_address('0.0.0.0')

  if not agglo_id:
      agglo_id = args.agglo_id

  split_seeds = split_seeds or args.split_seeds

  if split_seeds:
    split_seeds = load_split_seeds(split_seeds)

  splitter = InteractiveSplitter(
        graph, agglo_id=agglo_id, image_url=args.image_url,
        split_seeds=split_seeds, segmentation_url=args.segmentation_url)
  print(splitter.viewer)
  return splitter

def parse_args(argv=sys.argv[1:]):
  ap = argparse.ArgumentParser()

  ap.add_argument('-v', '--verbose', action='store_true',
                  help='Display verbose log messages.')

  common_ap = argparse.ArgumentParser(add_help=False)
  common_ap.add_argument(
      '--graph-db', required=True,
      help='Path to sqlite3 database specifying agglomeration graph')
  common_ap.add_argument('--image-url', required=True,
                         help='Neuroglancer data source URL for image')
  common_ap.add_argument('--segmentation-url', required=True,
                         help='Neuroglancer data source URL for segmentation')

  sub_aps = ap.add_subparsers(help='command to run')
  interactive_ap = sub_aps.add_parser('interactive', help='Interactively split an aglomerated component', parents=[common_ap])

  batch_ap = sub_aps.add_parser('batch', help='Split based on pre-specified seed files', parents=[common_ap])

  interactive_ap.add_argument('--agglo-id', type=int,
                              default=0,
                              help='Agglomerated component id to split')
  interactive_ap.add_argument('--split-seeds',
                              help='Path to JSON file specifying split seeds')
  interactive_ap.add_argument('--port', default=0)
  interactive_ap.set_defaults(func=run_interactive)

  batch_ap.add_argument('--split-seeds',
                        nargs='+',
                        help='Path to JSON file specifying split seeds')
  batch_ap.add_argument('--agglo-id', type=int,
                        help='Agglomerated component id to split')
  batch_ap.set_defaults(func=run_batch)

  args = ap.parse_args(argv)
  return args

def main():
  args = parse_args()
  graph = AgglomerationGraph(sqlite3.connect(args.graph_db, check_same_thread=False))

  if args.verbose:
    logging.basicConfig(level=logging.INFO)

  if args.agglo_id:
    args.func(args, graph)

  return (args, graph)

if __name__ == '__main__':
    main()

# python agglomeration_split_tool.py batch --graph ~/merge_graphs/274750196357:janelia-flyem-cx-flattened-tabs:sec24_seg_v2a:ffn_agglo_pass1_cpt5663627_medt160_with_celis_cx2-2048_r10_mask200_0.sqlite --image-url brainmaps://274750196357:janelia-flyem-cx-flattened-tabs:sec24_image --segmentation-url brainmaps://274750196357:janelia-flyem-cx-flattened-tabs:sec24_seg_v2a --split-seeds ~/tmp/celis_seeds_5e40_converted/*.json --html
# python split_interactive.py --graph-db ~/merge_log_sec24.sqlite --image-url brainmaps://274750196357:janelia-flyem-cx-flattened-tabs:sec24_image --segmentation-url brainmaps://274750196357:janelia-flyem-cx-flattened-tabs:sec24_seg_v2a --agglo-id 95668070
