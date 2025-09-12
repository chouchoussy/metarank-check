# run_online_metrics.py  (query-driven from metadata.json only)
import json, time, random, requests, argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

from online_metrics import aggregate, text_report

HOST = "http://localhost:9091"
MODEL = "models/xgboost"
NUM_QUERIES_DEFAULT = 7        # number of real queries to evaluate (sampled from mepython run_online_metrics.pytadata)
K_SEND_DEFAULT = 1000          # max items to send to /rank per query
QUERY_SIZE_DEFAULT = 400       # Top-N used to compute metrics per query (first page)
SEED_DEFAULT = 42
META_PATH_DEFAULT = "data/metadata.json"

def _extract_item_id(it):
    """Extract item id from a /rank response element."""
    if isinstance(it, str):
        return it
    if not isinstance(it, dict):
        return None
    # Metarank typically returns {"item": "..."}; some builds may return {"id": "..."}
    return it.get("id") or it.get("item") or it.get("identifier") or it.get("name")

def pick_candidates_for_query(q_items, k_send, META):
    """
    Pick candidate items for a real query (q_items).
    - De-duplicate while preserving original order
    - If more than k_send, take a random sample of size k_send
    - Reproducible via random.seed(...)
    """
    seen: Set[str] = set()
    uniq: List[str] = []
    for i in q_items:
        s = str(i)
        if s not in seen:
            seen.add(s)
            uniq.append(s)

    if k_send is None or k_send <= 0 or k_send >= len(uniq):
        return uniq  # take all

    return random.sample(uniq, k_send)

def fetch_rank_query(host, model, item_ids, meta, user="u_eval", session="s_eval"):
    """Call /rank for a single query and normalize the response into our metrics schema."""
    ts = int(time.time()*1000)
    payload = {
        "id": f"rq_eval_{ts}",
        "timestamp": ts,
        "user": user,
        "session": session,
        "items": [{"id": i} for i in item_ids]
    }
    r = requests.post(f"{host}/rank/{model}", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    items = data.get("items") or data.get("result") or data
    if not isinstance(items, list):
        raise RuntimeError(f"Unexpected /rank response: {str(data)[:400]}")

    out_items = []
    for pos, it in enumerate(items, start=1):
        iid = _extract_item_id(it)
        if not iid:
            raise KeyError(f"Cannot find item id in element: {items}")
        out_items.append({"id": iid, "position": pos, "attrs": meta.get(iid, {})})
    return {"timestamp": ts, "items": out_items}

def build_query_index_from_metadata(meta: Dict[str, dict]) -> Dict[str, List[str]]:
    """
    Build a map: query_id(str) -> [item_id,...] from metadata.json.
    - Normalize query_id to str to avoid issues like 5810.0 vs 5810
    - Skip items without query_id
    """
    q2items: Dict[str, List[str]] = defaultdict(list)
    for item_id, attrs in meta.items():
        if attrs is None:
            continue
        qid = attrs.get("query_id")
        if qid is None:
            continue
        qid_str = str(int(qid)) if isinstance(qid, float) and qid.is_integer() else str(qid)
        q2items[qid_str].append(str(item_id))
    return q2items

def parse_args():
    ap = argparse.ArgumentParser(description="Query-driven online evaluation using /rank and metadata.json")
    ap.add_argument("--host", default=HOST,
                    help="Metarank host, e.g. http://localhost:9090")
    ap.add_argument("--model", default=MODEL,
                    help="Model name as defined in config.yml (default: xgboost)")
    ap.add_argument("--num-queries", type=int, default=NUM_QUERIES_DEFAULT,
                    help="Number of real queries to evaluate (sampled from metadata.json)")
    ap.add_argument("--k-send", type=int, default=K_SEND_DEFAULT,
                    help="Max number of items to send to /rank for each query")
    ap.add_argument("--query-size", type=int, default=QUERY_SIZE_DEFAULT,
                    help="Top-N used to compute metrics per query (first page)")
    ap.add_argument("--meta", default=META_PATH_DEFAULT,
                    help="Path to data/metadata.json")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT,
                    help="Random seed for reproducible sampling")
    return ap.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    meta_path = Path(args.meta)
    if not meta_path.exists():
        raise SystemExit(f"Not found: {meta_path}. Please run the converter to create metadata.json first.")
    META: Dict[str, dict] = json.loads(meta_path.read_text())

    # 1) Build query -> items index from metadata.json
    q2items = build_query_index_from_metadata(META)
    if not q2items:
        raise SystemExit("metadata.json does not contain any query_id.")

    # 2) Sample N queries (reproducible by seed)
    all_qids = list(q2items.keys())
    sample_qids = random.sample(all_qids, k=min(args.num_queries, len(all_qids)))

    # 3) For each real query_id: take its items, pick candidates, call /rank
    queries = []
    for qid in sample_qids:
        items_of_q = q2items.get(qid, [])
        if not items_of_q:
            continue
        cands = pick_candidates_for_query(items_of_q, args.k_send, META)
        if not cands:
            continue
        queries.append(fetch_rank_query(args.host, args.model, cands, META))

    if not queries:
        raise SystemExit("No queries were collected from /rank.")

    # 4) Compute metrics on Top-N (first page) per query, then average across queries
    rep = aggregate(queries, query_size=args.query_size)
    print(text_report(rep))

if __name__ == "__main__":
    main()
