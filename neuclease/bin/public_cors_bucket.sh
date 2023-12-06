#!/bin/bash

set -x
set -e

gsutil iam ch allUsers:objectViewer ${1}

cat > /tmp/cors.json << EOF
[{"maxAgeSeconds": 3600, "method": ["GET"], "origin": ["*"], "responseHeader": ["Content-Type", "Range"]}]
EOF

gsutil cors set /tmp/cors.json ${1}
