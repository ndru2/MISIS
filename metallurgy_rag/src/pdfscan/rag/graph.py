"""Neo4j adapter: graph statements as ordinary retrieval candidates."""
from __future__ import annotations

import os
from hashlib import sha256

from pdfscan.rag import graph_core


def search(query: str, *, candidates=40, driver=None, depth=1) -> list[dict]:
    owned = driver is None
    if owned:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI', 'bolt://localhost:7690'),
            auth=(os.getenv('NEO4J_USER', 'neo4j'),
                  os.getenv('NEO4J_PASSWORD', 'neo4jpass')),
            connection_timeout=10, max_transaction_retry_time=15)
    try:
        with driver.session(database=os.getenv('NEO4J_DATABASE', 'neo4j')) as session:
            paths, seeds, _ = graph_core.search(
                session, query, depth=depth, paths_limit=candidates, use_vector=False)
            hits = {}
            for path in paths:
                for fact in path['facts']:
                    text = '\n'.join(filter(None, [graph_core.fact_line(fact),
                                       fact.get('sentence'), fact.get('table_text')]))
                    uid = str(fact.get('uid') or sha256(text.encode()).hexdigest())
                    key = 'kg:' + uid
                    if key in hits:
                        continue
                    page = fact.get('page')
                    hits[key] = {
                        'unit_id': key, 'unit_type': 'graph_statement',
                        'text': text, 'doc_id': fact.get('doc_id'),
                        'pages': [int(page)] if str(page).isdigit() else [],
                        'score': path['score'],
                        'graph': {'path': graph_core.path_line(path),
                                  'nodes': path['nodes'], 'relations': path['rels'],
                                  'sentence_uid': fact.get('sentence_uid'),
                                  'document_title': fact.get('doc_title'),
                                  'formulas': fact.get('formulas') or [],
                                  'seed_ids': [seed['id'] for seed in seeds]},
                    }
            return sorted(hits.values(), key=lambda hit: -hit['score'])[:candidates]
    finally:
        if owned:
            driver.close()
