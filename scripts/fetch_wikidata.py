#!/usr/bin/env python
"""Pull a small, fixed set of facts from Wikidata and cache them verbatim.

Every file records the SPARQL query and the date it was fetched, so any claim
built on it can be re-checked against the source. Nothing here is typed from
memory: if the endpoint is unreachable the script fails rather than inventing.

    python scripts/fetch_wikidata.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

ENDPOINT = "https://query.wikidata.org/sparql"
UA = "mint-research/0.1 (academic interpretability project)"
OUT = Path("data/wikidata")

QUERIES = {
    # narrow, concrete labels: one row per entity, one value per property
    "countries": """
SELECT ?entity (MIN(?capL) AS ?capital) (MIN(?langL) AS ?language)
       (MIN(?curL) AS ?currency) (MIN(?contL) AS ?continent) WHERE {
  ?c wdt:P31 wd:Q6256 ; wdt:P36 ?cap ; wdt:P37 ?lang ; wdt:P38 ?cur ; wdt:P30 ?cont .
  ?c rdfs:label ?entity   FILTER(lang(?entity)="en")
  ?cap rdfs:label ?capL   FILTER(lang(?capL)="en")
  ?lang rdfs:label ?langL FILTER(lang(?langL)="en")
  ?cur rdfs:label ?curL   FILTER(lang(?curL)="en")
  ?cont rdfs:label ?contL FILTER(lang(?contL)="en")
} GROUP BY ?entity ORDER BY ?entity LIMIT 60""",
    "country_extras": """
SELECT ?entity (MIN(?isoL) AS ?iso) (MIN(?callL) AS ?calling_code)
       (MIN(?sideL) AS ?driving_side) (MIN(?tldL) AS ?tld) WHERE {
  ?c wdt:P31 wd:Q6256 ; wdt:P297 ?isoL ; wdt:P474 ?callL ; wdt:P1622 ?side ; wdt:P78 ?tld .
  ?c rdfs:label ?entity   FILTER(lang(?entity)="en")
  ?side rdfs:label ?sideL FILTER(lang(?sideL)="en")
  ?tld rdfs:label ?tldL   FILTER(lang(?tldL)="en")
} GROUP BY ?entity ORDER BY ?entity LIMIT 200""",
    "cities": """
SELECT ?entity (MIN(?ctryL) AS ?country) (MIN(?contL) AS ?continent)
       (MIN(?langL) AS ?language) (MIN(?curL) AS ?currency) WHERE {
  { SELECT ?x WHERE { ?x wdt:P31 wd:Q1549591 } LIMIT 300 }
  ?x wdt:P17 ?ctry .
  ?ctry wdt:P30 ?cont ; wdt:P37 ?lang ; wdt:P38 ?cur .
  ?x rdfs:label ?entity   FILTER(lang(?entity)="en")
  ?ctry rdfs:label ?ctryL FILTER(lang(?ctryL)="en")
  ?cont rdfs:label ?contL FILTER(lang(?contL)="en")
  ?lang rdfs:label ?langL FILTER(lang(?langL)="en")
  ?cur rdfs:label ?curL   FILTER(lang(?curL)="en")
} GROUP BY ?entity ORDER BY ?entity LIMIT 60""",
}


def fetch(name: str, query: str) -> dict:
    url = f"{ENDPOINT}?format=json&query={urllib.parse.quote(query)}"
    r = subprocess.run(["curl", "-sS", "--max-time", "90", "-H", f"User-Agent: {UA}",
                        "-H", "Accept: application/sparql-results+json", url],
                       capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip().startswith("{"):
        raise SystemExit(f"{name}: Wikidata unreachable ({r.returncode}) "
                         f"{r.stderr[:200] or r.stdout[:200]}")
    raw = json.loads(r.stdout)
    rows = [{k: v["value"] for k, v in b.items()} for b in raw["results"]["bindings"]]
    return {"source": "Wikidata Query Service", "endpoint": ENDPOINT, "query": query.strip(),
            "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_rows": len(rows), "rows": rows}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, q in QUERIES.items():
        try:
            blob = fetch(name, q)
        except SystemExit as e:  # one unreachable query must not lose the others
            print(f"{name}: FAILED — {str(e)[:120]}")
            failed.append(name)
            continue
        (OUT / f"{name}.json").write_text(json.dumps(blob, indent=2, ensure_ascii=False))
        print(f"{name}: {blob['n_rows']} rows -> {OUT / (name + '.json')}")
        if blob["rows"]:
            print("   e.g.", blob["rows"][0])
    if failed:
        print(f"\nnot fetched: {failed} — rerun, or narrow the query")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
