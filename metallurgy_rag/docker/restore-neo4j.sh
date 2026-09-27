#!/bin/sh
set -eu
# Never overwrite an existing database, including a partially restored one.
if [ -d /data/databases/neo4j ]; then
  echo 'Existing Neo4j database found; restore skipped.'
  exit 0
fi
if [ ! -s /dump/neo4j.dump ]; then
  echo 'Missing /dump/neo4j.dump' >&2
  exit 1
fi
exec neo4j-admin database load neo4j --from-path=/dump
