#!/bin/sh

mkdir -p /app/redis
redis-server /app/conf/redis.conf &
PYTHON_UNBUFFERED=1 python -m py.delegator