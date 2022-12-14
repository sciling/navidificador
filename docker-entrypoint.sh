#! /bin/bash

set -eo pipefail
source /venv/bin/activate

if [ -z "$1" ]; then
    echo "Please, add a command to run this docker image"
    uvicorn bin.server:app --host 0.0.0.0 --port 8000
else
    "$@"
fi
