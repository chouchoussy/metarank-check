import json, time, random, hashlib, os
from pathlib import Path
import pandas as pd

IN_FILES = [
    "data/merged_data_train.ft",
    "data/merged_data_val.ft",
    "data/merged_data_test.ft",
]
OUT_DIR = Path("data")
EVENTS_PATH = OUT_DIR / "events.jsonl"
META_PATH = OUT_DIR / "metadata.json"

MAX_ITEMS_PER_QUERY = int(os.getenv("MAX_ITEMS_PER_QUERY", "1500"))
MAX_QUERIES         = int(os.getenv("MAX_QUERIES", "100"))  
RANDOM_SEED         = int(os.getenv("SEED", "42"))

NOW_MS = int(time.time()*1000)
DAY = 86_400_000
rnd = random.Random(RANDOM_SEED)

ACTION_MAP = {
    "click": "click",
    "view": "impression",
    "impression": "impression",
    "single_buy": "purchase",
    "imasia_buy": "purchase",
    "buy": "purchase",
    "purchase": "purchase",
    # "add_to_cart": "click"
    "no_action": None
}

def read_df(paths):
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        raise SystemExit("Not found any input Feather files.")
    return pd.concat([pd.read_feather(p) for p in paths], ignore_index=True)

def make_purchase_scalar(row):
    # add news field 'purchase' = item_single_sale_last_30_days + item_imasia_sale_last_30_days
    s30 = float(row.get("item_single_sale_last_30_days", 0) or 0)
    i30 = float(row.get("item_imasia_sale_last_30_days", 0) or 0)
    return s30 + i30

def build_metadata(df):
    meta = {}
    for row in df.itertuples(index=False):
        iid = str(getattr(row, "item_id"))
        attrs = {}
        for col in df.columns:
            if col == "item_id": 
                continue
            attrs[col] = getattr(row, col)
        # add 'purchase' field
        attrs["purchase"] = make_purchase_scalar(attrs)
        meta[iid] = attrs
    return meta

def make_item_event(item_id, attrs, ts):
    fields = [{"name": k, "value": v} for k, v in attrs.items() if v is not None]
    return {
        "event": "item",
        "id": f"it_{item_id}",
        "item": item_id,
        "timestamp": ts,
        "fields": fields
    }

def ranking_id_for_query(qid, ts, items):
    if qid is not None:
        return f"rk_{qid}"
    sig = f"{ts}_" + ",".join(sorted(items))
    return "rk_" + hashlib.sha1(sig.encode("utf-8")).hexdigest()[:12]

def make_ranking_event(user, session, qid, item_ids, ts):
    return {
        "event": "ranking",
        "id": ranking_id_for_query(qid, ts, item_ids),
        "timestamp": ts,
        "user": user,
        "session": session,
        "items": [{"id": x} for x in item_ids]
    }

def make_interaction_event(user, item_id, ts, itype, ranking_id=None):
    ev = {
        "event": "interaction",
        "id": f"in_{user}_{item_id}_{ts}",
        "timestamp": ts,
        "user": user,
        "item": item_id,
        "type": itype
    }
    if ranking_id:
        ev["ranking"] = ranking_id
    return ev

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_all = read_df(IN_FILES)

    # reuire columns
    for need in ("item_id",):
        if need not in df_all.columns:
            raise ValueError(f"Missing required column '{need}' in input data")

    # metadata (unique item_id)
    df_meta = df_all.drop_duplicates(subset=["item_id"]).reset_index(drop=True).copy()
    df_meta["item_id"] = df_meta["item_id"].astype(str)
    meta = build_metadata(df_meta)
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    events = []
    t0 = NOW_MS - 7*DAY
    ts_items = t0 - DAY

    # item events
    for iid, attrs in meta.items():
        events.append(make_item_event(iid, attrs, ts_items))

    # ranking + interaction
    have_query = "query_id" in df_all.columns
    have_action = "action" in df_all.columns
    have_user = "user_id" in df_all.columns
    have_auto = "auto_score" in df_all.columns
    have_bmhs = "bm25l_hs" in df_all.columns
    have_bm25 = "bm25l" in df_all.columns

    def choose_user(series):
        return str(series["user_id"]) if ("user_id" in series.index and pd.notna(series["user_id"])) else "u_anon"

    if have_query:
        df_all = df_all.dropna(subset=["query_id", "item_id"]).copy()
        df_all["query_id"] = df_all["query_id"].astype(str)
        df_all["item_id"]  = df_all["item_id"].astype(str)

        qids = df_all["query_id"].unique().tolist()
        if MAX_QUERIES:
            qids = qids[:MAX_QUERIES]

        for qi, qid in enumerate(qids, start=1):
            block = df_all[df_all["query_id"] == qid].copy()

            # sort candidate by auto_score > bm25l_hs > bm25l (desc)
            sort_cols = []
            if have_auto: sort_cols.append(("auto_score", False))
            if have_bmhs: sort_cols.append(("bm25l_hs", False))
            if have_bm25: sort_cols.append(("bm25l", False))
            if sort_cols:
                by = [c for c,_ in sort_cols]
                asc = [a for _,a in sort_cols]
                block = block.sort_values(by=by, ascending=asc)

            cands = block["item_id"].drop_duplicates().tolist()
            if MAX_ITEMS_PER_QUERY:
                cands = cands[:MAX_ITEMS_PER_QUERY]
            if not cands:
                continue

            user = choose_user(block.iloc[0]) if have_user else f"u_{qi%100:02d}"
            session = f"s_{qi%100:02d}"
            ts = t0 + qi*60_000

            rke = make_ranking_event(user, session, qid, cands, ts)
            events.append(rke)

            if have_action:
                clicked = block[block["item_id"].isin(cands)]
                for _, row in clicked.iterrows():
                    act = str(row["action"]).lower() if pd.notna(row["action"]) else ""
                    itype = ACTION_MAP.get(act)
                    if not itype:
                        continue
                    uid = str(row["user_id"]) if have_user and pd.notna(row["user_id"]) else user
                    its = ts + rnd.randint(1000, 10000)
                    events.append(make_interaction_event(uid, str(row["item_id"]), its, itype, rke["id"]))
    else:
        # no query_id column, make a single ranking of all items
        all_ids = df_meta["item_id"].astype(str).tolist()
        if MAX_ITEMS_PER_QUERY:
            all_ids = all_ids[:MAX_ITEMS_PER_QUERY]
        ts = t0 + 60_000
        events.append(make_ranking_event("u_anon","s_anon", None, all_ids, ts))

    # sort by timestamp and write out
    events.sort(key=lambda e: int(e.get("timestamp", 0)))
    with EVENTS_PATH.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    print(f"Wrote {len(events)} events to {EVENTS_PATH}")
    print(f"Wrote metadata to {META_PATH}")

if __name__ == "__main__":
    main()
