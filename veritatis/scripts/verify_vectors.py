#!/usr/bin/env python
"""Verify that Milvus actually stores vectors after /ingest.

Examples:
  python scripts/verify_vectors.py --collection veritatis_tier1_lake --limit 3
  python scripts/verify_vectors.py --collection veritatis_tier1_lake --id <record_id> --show-embedding
  python scripts/verify_vectors.py --collection veritatis_tier1_lake --search "earthquake near Tokyo" --topk 5

Checks performed:
  - Connect to Milvus (host/port from env MILVUS_HOST/MILVUS_PORT).
  - Confirm collection exists; print schema & index summary.
  - Count entities.
  - Fetch one or more records via MilvusClient.get (ensures embeddings are returned).
  - Compute vector length & L2 norm for sanity (norm ~1 if normalized).
  - Optional semantic search: generates embedding with local model (requires model downloaded).
Outputs JSON summary to stdout.
"""
import os, sys, json, math, argparse
from statistics import mean
from pymilvus import connections, utility, Collection, MilvusClient

DEF_HOST = os.getenv("MILVUS_HOST", "localhost")
DEF_PORT = os.getenv("MILVUS_PORT", "19530")

try:
    # Attempt import of embedding generator (may trigger large model download if first run)
    from veritatis.embeddings import embedding_generator
    HAVE_EMBED = True
except Exception as e:  # pragma: no cover
    HAVE_EMBED = False
    EMBED_IMPORT_ERROR = str(e)


def connect():
    connections.connect(host=DEF_HOST, port=DEF_PORT)


def fetch(client: MilvusClient, collection: str, ids):
    return client.get(collection, ids)


def l2_norm(vec):
    return math.sqrt(sum(v * v for v in vec))


def summarize_vectors(rows):
    dims = [len(r.get("embedding", [])) for r in rows if "embedding" in r]
    norms = [l2_norm(r.get("embedding", [])) for r in rows if "embedding" in r]
    return {
        "count": len(rows),
        "dims": dims,
        "unique_dims": sorted(set(dims)),
        "norms": norms,
        "mean_norm": round(mean(norms), 6) if norms else None,
    }


def do_search(collection_name: str, query: str, topk: int):
    if not HAVE_EMBED:
        return {"error": "embedding_generator_not_available", "detail": EMBED_IMPORT_ERROR}
    emb = embedding_generator.generate(query)
    col = Collection(collection_name)
    col.load()
    res = col.search(data=[emb], anns_field="embedding", param={"metric_type": "COSINE"}, limit=topk, output_fields=["id","content"])
    out = []
    for hits in res:
        for hit in hits:
            out.append({
                "id": hit.id,
                "distance": hit.distance,
                "content": hit.entity.get("content")
            })
    return {"query": query, "topk": topk, "results": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--id", help="Specific record id to fetch")
    ap.add_argument("--limit", type=int, default=3, help="Sample N records if --id not provided")
    ap.add_argument("--show-embedding", action="store_true", help="Include raw embedding arrays in output")
    ap.add_argument("--search", help="Text query to perform similarity search")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    connect()

    if not utility.has_collection(args.collection):
        print(json.dumps({"error": "collection_not_found", "collection": args.collection}, indent=2))
        return

    col = Collection(args.collection)
    schema_fields = [f.name for f in col.schema.fields]
    # entity count may require flush
    col.load()
    count = col.num_entities

    client = MilvusClient(uri=f"http://{DEF_HOST}:{DEF_PORT}")

    rows = []
    if args.id:
        got = fetch(client, args.collection, [args.id])
        rows = got if got else []
    else:
        # Get some IDs via a broad query (may be slow if huge collection)
        sample = col.query(expr="id != ''", output_fields=["id"], limit=args.limit)
        for r in sample:
            rid = r["id"]
            full = fetch(client, args.collection, [rid])
            if full:
                rows.append(full[0])

    vector_summary = summarize_vectors(rows)

    # Optionally strip embeddings for readability
    if not args.show_embedding:
        for r in rows:
            if "embedding" in r:
                r["embedding_dim"] = len(r["embedding"])
                del r["embedding"]

    search_res = None
    if args.search:
        search_res = do_search(args.collection, args.search, args.topk)

    out = {
        "collection": args.collection,
        "schema_fields": schema_fields,
        "entity_count": count,
        "fetched_rows": rows,
        "vector_summary": vector_summary,
        "search": search_res,
        "embedding_module_loaded": HAVE_EMBED,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
