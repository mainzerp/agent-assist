#!/bin/sh
# Fix ownership of volume data (may be root-owned from previous images)
chown -R app:app /data 2>/dev/null || true
exec gosu app "$@"
