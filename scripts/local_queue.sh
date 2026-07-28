#!/bin/bash
# Generic name for the local lane runner. Usage:
#   scripts/local_queue.sh <queue-file>
exec "$(dirname "$0")/pi_queue.sh" "$@"
