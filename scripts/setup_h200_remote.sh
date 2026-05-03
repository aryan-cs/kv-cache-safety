#!/usr/bin/env bash
set -euo pipefail

host="${H200_HOST:-uiuc-h200}"

ssh "$host" 'bash -s' < "$(dirname "$0")/bootstrap_h200.sh"
