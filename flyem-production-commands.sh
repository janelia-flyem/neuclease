#!/bin/bash

#
# Example launch commands for FlyEM cleave server instances
#

ACTIVATE=/groups/flyem/proj/cluster/miniforge/bin/activate
source $ACTIVATE flyem

# emdata3:8900 -- 55ce82b0567b4987960652a169f9b7ff
# Run this on emdata3 to produce a cleave server on emdata3:5551

nohup neuclease_cleave_server \
  -p 5551 \
  --log-dir /nrs/flyem/bergs/neuclease-logs/production-emdata3-8900 \
  --merge-table /nrs/flyem/bergs/final-agglo-fixsplit-patched/final_patched_20180426_merge_table.npy \
  --primary-dvid-server emdata3:8900 \
  --primary-uuid 55ce82b0567b4987960652a169f9b7ff \
  --primary-labelmap-instance segmentation \
  --split-mapping /nrs/flyem/bergs/final-agglo-fixsplit-patched/split-lineage-9ec0b3.csv \
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
