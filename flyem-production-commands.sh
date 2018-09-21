
#!/bin/bash

# Example launch commands for FlyEM cleave server instances
#

ACTIVATE=/groups/flyem/proj/cluster/miniforge/bin/activate
source $ACTIVATE flyem

# emdata3:8900 -- d5852c27b5c04687bb1be414f6dc2336
# Run this on emdata3 to produce a cleave server on emdata3:5551

CLEAVE_PORT=5551
nohup neuclease_cleave_server \
  -p ${CLEAVE_PORT} \
  --log-dir /nrs/flyem/bergs/neuclease-logs/production-emdata3-8900-${CLEAVE_PORT} \
  --merge-table /nrs/flyem/bergs/final-agglo-fixsplit-patched/final_patched_20180426_merge_table.npy \
  --primary-dvid-server emdata3:8900 \
  --primary-uuid 07160ccb9ee849ad8465c3b617bb90e5 \
  --primary-labelmap-instance segmentation \
  &
##

