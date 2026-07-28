#!/bin/bash
# Generic name for the remote lane runner. Usage:
#   scripts/remote_queue.sh <queue-file> [host]
exec "$(dirname "$0")/cse_queue.sh" "$@"
