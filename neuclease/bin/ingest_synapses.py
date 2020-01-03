"""
Example script for ingesting synapses into a DVID annotation instance
and syncing it to a DVID segmentation instance.

(See the code comments in ingest_synapses.py for more details.)

Here, we assume you have already prepared a JSON file in DVID's
expected format (see dvid's /api/help/annotation page for details),
and you already have your segmentation loaded into a 'labelmap' instance,
but you haven't yet created the 'annotation' instance to store the synapses.

DVID expects synapses in a JSON list, in the same format as that returned by the /elements endpoint:

[
    {"Pos":[5750,20489,20542], "Kind":"PostSyn", "Tags":[], "Prop":{"conf":"0.473165","user":"$fpl"}, "Rels":[{"Rel":"PostSynTo","To":[5751,20450,20556]}]},
    {"Pos":[5940,20510,20517], "Kind":"PostSyn", "Tags":[], "Prop":{"conf":"0.594747","user":"$fpl"}, "Rels":[{"Rel":"PostSynTo","To":[5941,20486,20536]}]},
    {"Pos":[5951,20504,20540], "Kind":"PostSyn", "Tags":[], "Prop":{"conf":"0.741894","user":"$fpl"}, "Rels":[{"Rel":"PostSynTo","To":[5941,20486,20536]}]},
    {"Pos":[5941,20486,20536], "Kind":"PreSyn", "Tags":[], "Prop":{"conf":"0.99","user":"$fpl"},
        "Rels":[
            {"Rel":"PreSynTo","To":[5952,20499,20517]},{"Rel":"PreSynTo","To":[5960,20466,20517]},
            {"Rel":"PreSynTo","To":[5963,20490,20517]},{"Rel":"PreSynTo","To":[5940,20510,20517]},
            {"Rel":"PreSynTo","To":[5967,20484,20540]},{"Rel":"PreSynTo","To":[5951,20504,20540]},
            {"Rel":"PreSynTo","To":[5973,20453,20553]}]
    },
    {"Pos":[5952,20499,20517], "Kind":"PostSyn", "Tags":[], "Prop":{"conf":"0.858438","user":"$fpl"}, "Rels":[{"Rel":"PostSynTo","To":[5941,20486,20536]}]},
]
"""
import argparse

import ujson
import pandas as pd

from neuclease.util import tqdm_proxy
from neuclease.dvid import create_instance, post_elements, post_sync, post_reload

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--annotation-instance', default='synapses')
    parser.add_argument('--labelmap-instance', default='segmentation')
    parser.add_argument('--labelsz-instance')
    parser.add_argument('server')
    parser.add_argument('uuid')
    parser.add_argument('elements_json')

    args = parser.parse_args()

    server = args.server
    uuid = args.uuid
    syn_instance = args.annotation_instance
    seg_instance = args.labelmap_instance

    ##
    ## 1. Create an 'annotation' instance to store the synapse data
    ##
    ##      POST .../instance
    ## 
    create_instance(server, uuid, syn_instance, 'annotation')

    ##
    ## 2. Upload the synapse elements.
    ##
    ##      POST .../elements
    ##
    ##    Note:
    ##      DVID stores these in block-aligned groups, based on the synapse coordinates.
    ##      Ingestion will be fastest if you pre-sort your JSON elements by 64px blocks,
    ##      in Z-Y-X order, as shown below.
    ##

    with open(args.elements_json, 'r') as f:
        elements = ujson.load(f)
    
    # Sort elements by block location (64px blocks)
    # FIXME: This code should work but I haven't tested it yet.  Might have a typo.
    elements_df = pd.DataFrame([(*e["Pos"], e) for e in elements], columns=['x', 'y', 'z', 'element'])
    elements_df[['z', 'y', 'x']] //= 64
    elements_df.sort_values(['z', 'y', 'x'], inplace=True)
    
    # Group blocks into larger chunks, with each chunk being 100 blocks
    # in the X direction (and 1x1 in the zy directions).
    elements_df['x'] //= 100

    # Ingest in chunks.
    num_chunks = elements_df[['z', 'y', 'x']].drop_duplicates().shape[0]
    chunked_df = elements_df.groupby(['z', 'y', 'x'])
    for _zyx, batch_elements_df in tqdm_proxy(chunked_df, total=num_chunks):
        post_elements(server, uuid, syn_instance, batch_elements_df['element'].tolist())
    
    ##
    ## 3. Sync the annotation instance to a pre-existing
    ##    segmentation (labelmap) instance.
    ##
    ##      POST .../sync
    ##
    ##    This 'subscribes' the annotation instance to changes in the segmentation,
    ##    keeping updated counts of synapses in each body.
    ##    This will enable the .../<annotation>/labels endpoint to work efficiently.
    ##
    post_sync(server, uuid, syn_instance, [seg_instance])
    
    ##
    ## 4. Reload the synapse instance AFTER the sync was configured (above).
    ##    For real-world data sizes (e.g. millions of synapses) this will take
    ##    a long time (hours).
    ##
    ##      POST .../reload
    ##
    post_reload(server, uuid, syn_instance)

    ##
    ## 5. (Optional)
    ##    For some proofreading protocols, you may wish to create a 'labelsz' (label size) instance,
    ##    which allows you to ask for the largest N bodies (by synapse count).
    ##
    ##
    if args.labelsz_instance:
        create_instance(server, uuid, args.labelsz_instance, 'labelsz')
        post_sync(server, uuid, args.labelsz_instance, [syn_instance])
        post_reload(server, uuid, args.labelsz_instance)
    
    
if __name__ == "__main__":
    main()
