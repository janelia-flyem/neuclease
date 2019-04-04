
#!/bin/bash

# Example launch commands for FlyEM cleave server instances
#

ACTIVATE=/groups/flyem/proj/cluster/miniforge/bin/activate
source $ACTIVATE flyem

# Using emdata3:8900 as the primary node, cleave server hosted on emdata1
# Run this on emdata3 to produce a cleave server on emdata1:5553

CLEAVE_PORT=5551
nohup neuclease_cleave_server \
  -p ${CLEAVE_PORT} \
  --log-dir /groups/flyem/proj/cleave-files/logs/production-emdata4-8900-emdata3-${CLEAVE_PORT} \
  --merge-table /groups/flyem/proj/cleave-files/merge-tables/merge_table_20180426_updated_to_28841c_with_frankenbody_svs.npy \
  --primary-dvid-server emdata4:8900 \
  --primary-uuid 415836cb8bee4d23ab7c9dbf5c5bb9c9 \
  --primary-labelmap-instance segmentation \
  &
##

