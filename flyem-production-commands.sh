#!/bin/bash

#
# Example launch commands for FlyEM cleave server instances
#

ACTIVATE=/groups/flyem/proj/cluster/miniforge/bin/activate
source $ACTIVATE flyem

# emdata3:8900 -- 039784741703407ea25c9acdc6d0db8c
# Run this on emdata3 to produce a cleave server on emdata3:5551

nohup neuclease_cleave_server \
  -p 5551 \
  --log-dir /nrs/flyem/bergs/neuclease-logs/production-emdata3-8900 \
  --merge-table /nrs/flyem/bergs/final-agglo-fixsplit-patched/final_patched_20180426_merge_table.npy \
  --primary-dvid-server emdata3:8900 \
  --primary-uuid 039784741703407ea25c9acdc6d0db8c \
  --primary-labelmap-instance segmentation \
  --split-mapping /nrs/flyem/bergs/final-agglo-fixsplit-patched/split-lineage-017ad.csv \
  &
##


##
## Practice server (same as above, but with a different dvid server and different log directory)
##
PRACTICE_PARENT_UUID="6134ca01a0cf444baf82d5bc1efb49e8"
nohup neuclease_cleave_server \
  -p 5551 \
  --log-dir /nrs/flyem/bergs/neuclease-logs/practice-emdata1-8900 \
  --merge-table /nrs/flyem/bergs/final-agglo-fixsplit-patched/final_patched_20180426_merge_table.npy \
  --primary-dvid-server emdata1:8900 \
  --primary-uuid ${PRACTICE_PARENT_UUID} \
  --primary-labelmap-instance segmentation \
  --split-mapping /nrs/flyem/bergs/final-agglo-fixsplit-patched/split-lineage-017ad.csv \
  &
##
