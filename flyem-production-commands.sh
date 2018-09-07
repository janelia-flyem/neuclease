
#!/bin/bash

# Example launch commands for FlyEM cleave server instances
#

ACTIVATE=/groups/flyem/proj/cluster/miniforge/bin/activate
source $ACTIVATE flyem

# emdata3:8900 -- d5852c27b5c04687bb1be414f6dc2336
# Run this on emdata3 to produce a cleave server on emdata3:5551

CLEAVE_PORT=5551
nohup neuclease_cleave_server \
  -p 5551 \
  --log-dir /nrs/flyem/bergs/neuclease-logs/production-emdata3-8900-${CLEAVE_PORT} \
  --merge-table /nrs/flyem/bergs/final-agglo-fixsplit-patched/final_patched_20180426_merge_table.npy \
  --primary-dvid-server emdata3:8900 \
  --primary-uuid 9e0d2e899d624d47a53602f3ae986633 \
  --primary-labelmap-instance segmentation \
  &
##

CLEAVE_PORT=5552
nohup neuclease_cleave_server \
  -p ${CLEAVE_PORT} \
  --log-dir /nrs/flyem/bergs/neuclease-logs/production-emdata3-8900-${CLEAVE_PORT} \
  --merge-table /nrs/flyem/bergs/final-agglo-fixsplit-patched/final_patched_20180426_merge_table.npy \
  --primary-dvid-server emdata3:8900 \
  --primary-uuid 9e0d2e899d624d47a53602f3ae986633 \
  --primary-labelmap-instance segmentation \
  &
##

##
## DEVELOPMENT (port 5552)
##
nohup neuclease_cleave_server \
  -p 5552 \
  --log-dir /nrs/flyem/bergs/neuclease-logs/development-emdata1-8400 \
  --merge-table /nrs/flyem/bergs/final-agglo-fixsplit-patched/final_patched_20180426_merge_table.npy \
  --primary-dvid-server emdata1:8400 \
  --primary-uuid ecbedddd08034db8b0b0f1529b578d96 \
  --primary-labelmap-instance segmentation \
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
  &
##
