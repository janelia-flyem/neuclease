#!/bin/bash

set -e

GRAYSCALE_VOLUME_ID="274750196357:janelia-flyem-cx-flattened-tabs:sec24_image"
BASE_SEGMENTATION_VOLUME_ID="274750196357:janelia-flyem-cx-flattened-tabs:sec24_seg_v2a"
CHANGE_STACK_ID="ffn_agglo_pass1_cpt5663627_medt160_with_celis_cx2-2048_r10_mask200_0"
MERGE_GRAPH_DB="${BASE_SEGMENTATION_VOLUME_ID}:${CHANGE_STACK_ID}.sqlite"

if [ ! -e exported_merge_graphs/${MERGE_GRAPH_DB} ]; then
    # Download the merge graph in sqlite format.
    mkdir -p exported_merge_graphs
    gsutil cp "gs://janelia-cx/exported_merge_graphs/${MERGE_GRAPH_DB}" exported_merge_graphs/

	# Download the converted split seed JSON files.
	#gsutil cp -r gs://janelia-cx/celis_seeds_5e40_converted .
fi

#BODY_ID_TO_SPLIT=107017171
#BODY_ID_TO_SPLIT=21497262
#BODY_ID_TO_SPLIT=32659079
#BODY_ID_TO_SPLIT=37960616
#BODY_ID_TO_SPLIT=50291899
#BODY_ID_TO_SPLIT=59096911
#BODY_ID_TO_SPLIT=61034789
#BODY_ID_TO_SPLIT=83994412
#BODY_ID_TO_SPLIT=95668070
#
#
#echo "To split a body, try: "
#echo ">>> run_interactive(args, graph, ${BODY_ID_TO_SPLIT})"
#
## It is important to run python -i, rather than python,
## so that python stays running while you use the tool.
#python -i agglomeration_split_tool.py \
#    interactive \
#    --graph exported_merge_graphs/${MERGE_GRAPH_DB} \
#    --image-url brainmaps://${GRAYSCALE_VOLUME_ID} \
#    --segmentation-url brainmaps://${BASE_SEGMENTATION_VOLUME_ID} \
##

python viewer_list_server.py \
    interactive \
    --graph exported_merge_graphs/${MERGE_GRAPH_DB} \
    --image-url brainmaps://${GRAYSCALE_VOLUME_ID} \
    --segmentation-url brainmaps://${BASE_SEGMENTATION_VOLUME_ID}
##

