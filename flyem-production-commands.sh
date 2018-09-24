
#!/bin/bash

# Example launch commands for FlyEM cleave server instances
#

ACTIVATE=/groups/flyem/proj/cluster/miniforge/bin/activate
source $ACTIVATE flyem


# Using emdata3:8900 as the primary node, cleave server hosted on emdata1
# Run this on emdata1 to produce a cleave server on emdata1:5551

CLEAVE_PORT=5551
nohup neuclease_cleave_server \
  -p ${CLEAVE_PORT} \
  --log-dir /nrs/flyem/bergs/neuclease-logs/production-emdata3-8900-emdata1-${CLEAVE_PORT} \
  --merge-table /nrs/flyem/bergs/final-agglo-fixsplit-patched/final_patched_20180426_merge_table.npy \
  --primary-dvid-server emdata3:8900 \
  --primary-uuid 7f0cf466136b4d56aeaeffe4a494c6ab \
  --primary-labelmap-instance segmentation \
  &
##

