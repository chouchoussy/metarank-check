from __future__ import annotations
from typing import List, Dict, Any, Optional, Iterable
from collections import Counter
import math
import statistics

def _parse_bool(v):
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1","true","yes","y","t"): return True
        if s in ("0","false","no","n","f"): return False
    return None

def _is_num(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))

def _safe_mean(xs: Iterable[Optional[float]]) -> float:
    xs = [x for x in xs if _is_num(x)]
    return sum(xs)/len(xs) if xs else float('nan')

def _safe_median(xs: Iterable[Optional[float]]) -> float:
    xs = [x for x in xs if _is_num(x)]
    return statistics.median(xs) if xs else float('nan')

def _safe_std(xs: Iterable[Optional[float]]) -> float:
    xs = [x for x in xs if _is_num(x)]
    if len(xs) < 2: return float('nan')
    try:
        return statistics.pstdev(xs)
    except Exception:
        return float('nan')

def _norm_entropy(labels: list) -> float:
    n = len(labels)
    if n == 0: return float('nan')
    counts = Counter(labels)
    ps = [c/n for c in counts.values()]
    H = -sum(p*math.log(p, 2) for p in ps if p > 0)
    Hmax = math.log(len(counts), 2) if len(counts) > 1 else 1.0
    return (H/Hmax) if Hmax > 0 else 0.0

def _compute_topk_share(flags: list[bool], positions: list[int], ks: list[int]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if not flags or not positions:
        return {k: float('nan') for k in ks}
    for k in ks:
        num = sum(1 for f, p in zip(flags, positions) if f and isinstance(p, int) and 1 <= p <= k)
        den = sum(1 for p in positions if isinstance(p, int) and 1 <= p <= k)
        out[k] = (num/den) if den > 0 else float('nan')
    return out

def aggregate(queries: List[Dict[str, Any]], query_size: int = 200, **kwargs) -> Dict[str, Any]:
    """
    Compute all metrics over a list of queries (each query = one /rank result list).

    Args:
        queries: list of {"timestamp": int|None, "items": [{"id","position","attrs"}...]}
        query_size: Top-N items per query used for evaluation (first page).
        **kwargs: supports alias 'page_size' to keep backward compatibility.

    Returns:
        Dict of aggregated metrics across queries.
    """
    # Backward-compat: allow page_size alias
    if "page_size" in kwargs and isinstance(kwargs["page_size"], int):
        query_size = kwargs["page_size"]

    if not queries:
        return {
            "queries_evaluated": 0,
            "query_size": query_size,
            "coverage": {},
            "exclusive": {},
            "freshness_days": {},
            "diversity": {},
            "photo_ai": {},
            "sales": {}
        }

    items_per_query = []
    cov_total = 0
    cov_has_exclusive = 0
    cov_has_contrib_id = 0
    cov_has_ai_flag = 0
    cov_has_content_type = 0
    cov_has_time_on_sales = 0
    cov_has_purchase = 0

    # exclusive
    exclusive_first_pos = []
    exclusive_last_pos  = []
    exclusive_mean_pos  = []
    exclusive_share     = []
    exclusive_unique_contrib_counts = []
    exclusive_topk_per_query: List[Dict[int, float]] = []
    topk_list = [5, 10, 20, 50]

    # freshness (time_on_sales: days)
    latest_days_list = []
    oldest_days_list = []
    mean_days_list   = []
    median_days_list = []
    std_days_list    = []
    cold_7_share_list  = []
    cold_30_share_list = []

    # diversity
    contrib_diversities = []
    content_diversities = []
    unique_contrib_counts = []
    unique_content_counts = []

    # photo ai
    ai_rates = []
    non_ai_rates = []
    unknown_ai_rates = []

    # purchase-based ranking signals
    purch_rank_avg = []
    purch_rank_med = []
    sale_topn_thresholds = [10, 50, 100, 200]
    purch_topn_multi: Dict[int, list] = {k: [] for k in sale_topn_thresholds}

    for q in queries:
        items = q.get("items", [])
        n_on_query = min(query_size, len(items))
        items_per_query.append(n_on_query)

        positions_exclusive = []
        exclusive_contributors = set()

        pos_list = []
        exclusive_flags = []
        time_on_sales_vals = []
        contrib_ids = []
        content_types = []
        ai_flags = []
        purchase_vals = []

        cold_denom = 0
        cold_le7 = 0
        cold_le30 = 0

        for it in items[:query_size]:
            pos = it.get("position")
            pos_list.append(pos if isinstance(pos, int) else None)
            attrs = it.get("attrs", {}) or {}

            cov_total += 1
            if "exclusive_contributor" in attrs: cov_has_exclusive += 1
            if "contributor_id" in attrs and attrs.get("contributor_id") not in (None, "", "nan"): cov_has_contrib_id += 1
            if "item_is_ai_generated" in attrs: cov_has_ai_flag += 1
            if "content_type" in attrs: cov_has_content_type += 1
            if "time_on_sales" in attrs and _is_num(attrs.get("time_on_sales")): cov_has_time_on_sales += 1
            if "purchase" in attrs and _is_num(attrs.get("purchase")): cov_has_purchase += 1

            # exclusive
            is_excl = _parse_bool(attrs.get("exclusive_contributor"))
            exclusive_flags.append(is_excl is True)
            if is_excl is True:
                if isinstance(pos, int) and pos >= 1:
                    positions_exclusive.append(pos)
                cid = attrs.get("contributor_id")
                if cid: exclusive_contributors.add(cid)

            # freshness
            tos = attrs.get("time_on_sales")
            if _is_num(tos):
                d = float(tos)
                time_on_sales_vals.append(d)
                cold_denom += 1
                if d <= 7:  cold_le7  += 1
                if d <= 30: cold_le30 += 1

            # diversity pools
            cid = attrs.get("contributor_id")
            if cid: contrib_ids.append(cid)
            ct = attrs.get("content_type")
            if ct is not None: content_types.append(ct)

            # AI flags
            ai = attrs.get("item_is_ai_generated")
            if ai is None:
                ai_flags.append(None)
            else:
                try:
                    ai_flags.append(bool(float(ai) > 0.5))
                except Exception:
                    ai_flags.append(None)

            # purchase value
            pv = attrs.get("purchase")
            purchase_vals.append(float(pv) if _is_num(pv) else None)

        # exclusive per query
        if positions_exclusive:
            exclusive_first_pos.append(min(positions_exclusive))
            exclusive_last_pos.append(max(positions_exclusive))
            exclusive_mean_pos.append(sum(positions_exclusive)/len(positions_exclusive))
        else:
            exclusive_first_pos.append(float('nan'))
            exclusive_last_pos.append(float('nan'))
            exclusive_mean_pos.append(float('nan'))

        denom_q = max(1, n_on_query)
        exclusive_share.append(sum(1 for e in exclusive_flags if e) / denom_q)
        exclusive_unique_contrib_counts.append(len(exclusive_contributors))
        exclusive_topk_per_query.append(_compute_topk_share(exclusive_flags, pos_list, [k for k in topk_list if k <= query_size]))

        # freshness per query
        latest_days_list.append(min(time_on_sales_vals) if time_on_sales_vals else float('nan'))
        oldest_days_list.append(max(time_on_sales_vals) if time_on_sales_vals else float('nan'))
        mean_days_list.append(_safe_mean(time_on_sales_vals))
        median_days_list.append(_safe_median(time_on_sales_vals))
        std_days_list.append(_safe_std(time_on_sales_vals))
        if cold_denom > 0:
            cold_7_share_list.append(cold_le7/cold_denom)
            cold_30_share_list.append(cold_le30/cold_denom)
        else:
            cold_7_share_list.append(float('nan'))
            cold_30_share_list.append(float('nan'))

        # diversity
        contrib_diversities.append(_norm_entropy(contrib_ids))
        content_diversities.append(_norm_entropy(content_types))
        unique_contrib_counts.append(len(set(contrib_ids)))
        unique_content_counts.append(len(set(content_types)))

        # photo AI per query
        ai_count = sum(1 for f in ai_flags if f is True)
        non_ai_count = sum(1 for f in ai_flags if f is False)
        unknown_count = sum(1 for f in ai_flags if f is None)
        denom_ai = max(1, len(ai_flags))
        ai_rates.append(ai_count/denom_ai)
        non_ai_rates.append(non_ai_count/denom_ai)
        unknown_ai_rates.append(unknown_count/denom_ai)

        # purchase rank (smaller rank = better)
        valid = [(i, v) for i, v in enumerate(purchase_vals) if _is_num(v)]
        if valid:
            sorted_idx = [i for i, _ in sorted(valid, key=lambda t: (-t[1], t[0]))]
            ranks = [None] * len(purchase_vals)
            for r, idx in enumerate(sorted_idx, start=1):
                ranks[idx] = r
            r_valid = [r for r in ranks if _is_num(r)]
            purch_rank_avg.append(_safe_mean(r_valid))
            purch_rank_med.append(_safe_median(r_valid))
            for th in sale_topn_thresholds:
                if th <= query_size:
                    purch_topn_multi[th].append(sum(1 for r in r_valid if r is not None and r <= th) / denom_q)
        else:
            purch_rank_avg.append(float('nan'))
            purch_rank_med.append(float('nan'))
            for th in sale_topn_thresholds:
                purch_topn_multi[th].append(float('nan'))

    # aggregate exclusive top-k
    excl_topk_avg: Dict[int, float] = {}
    if exclusive_topk_per_query:
        ks = set().union(*[d.keys() for d in exclusive_topk_per_query])
        for k in ks:
            excl_topk_avg[k] = _safe_mean([d.get(k) for d in exclusive_topk_per_query])

    coverage = {
        "avg_items_per_query": _safe_mean(items_per_query),
        "fields_presence": {
            "exclusive_contributor": (cov_has_exclusive / cov_total) if cov_total else float('nan'),
            "contributor_id":        (cov_has_contrib_id / cov_total) if cov_total else float('nan'),
            "item_is_ai_generated":  (cov_has_ai_flag / cov_total)    if cov_total else float('nan'),
            "content_type":          (cov_has_content_type / cov_total) if cov_total else float('nan'),
            "time_on_sales":         (cov_has_time_on_sales / cov_total) if cov_total else float('nan'),
            "purchase":              (cov_has_purchase / cov_total)   if cov_total else float('nan'),
        }
    }

    report = {
        "queries_evaluated": len(queries),
        "query_size": query_size,
        "coverage": coverage,
        "exclusive": {
            "proportion_items": _safe_mean(exclusive_share),
            "unique_contributors": _safe_mean(exclusive_unique_contrib_counts),
            "avg_first_rank": _safe_mean(exclusive_first_pos),
            "avg_last_rank":  _safe_mean(exclusive_last_pos),
            "avg_mean_rank":  _safe_mean(exclusive_mean_pos),
            "topk_share":     excl_topk_avg
        },
        "freshness_days": {
            "avg_latest": _safe_mean(latest_days_list),
            "avg_oldest": _safe_mean(oldest_days_list),
            "avg_mean":   _safe_mean(mean_days_list),
            "median_mean": _safe_median(mean_days_list),
            "std_mean":    _safe_std(mean_days_list),
            "coldstart_le_7_share":  _safe_mean(cold_7_share_list),
            "coldstart_le_30_share": _safe_mean(cold_30_share_list),
        },
        "diversity": {
            "contributors_normalized_entropy": _safe_mean(contrib_diversities),
            "content_normalized_entropy":      _safe_mean(content_diversities),
            "unique_contributors_avg": _safe_mean(unique_contrib_counts),
            "unique_content_avg":      _safe_mean(unique_content_counts),
        },
        "photo_ai": {
            "ai_impression_rate":      _safe_mean(ai_rates),
            "non_ai_impression_rate":  _safe_mean(non_ai_rates),
            "unknown_impression_rate": _safe_mean(unknown_ai_rates),
        },
        "sales": {
            "avg_purchase_rank":    _safe_mean(purch_rank_avg),
            "median_purchase_rank": _safe_median(purch_rank_med),
            "topn_shares":          {th: _safe_mean(vs) for th, vs in purch_topn_multi.items()},
        }
    }
    return report

def text_report(metrics: Dict[str, Any]) -> str:
    def pct(x):
        if x is None or (isinstance(x, float) and math.isnan(x)): return "NA"
        return f"{x*100:.2f}%"
    def num(x, nd=2):
        if x is None or (isinstance(x, float) and math.isnan(x)): return "NA"
        return f"{x:.{nd}f}"

    cov = metrics.get("coverage", {})
    ex  = metrics.get("exclusive", {})
    fr  = metrics.get("freshness_days", {})
    dv  = metrics.get("diversity", {})
    ph  = metrics.get("photo_ai", {})
    sa  = metrics.get("sales", {})

    lines = []
    lines.append(f"Online evaluation (queries={metrics.get('queries_evaluated',0)}, query_size={metrics.get('query_size','NA')}):")

    fp = cov.get("fields_presence", {})
    lines.append("  • Coverage:")
    lines.append(f"      – Avg items per query: {num(cov.get('avg_items_per_query'))}")
    lines.append(f"      – Fields presence (overall):")
    lines.append(f"          · exclusive_contributor: {pct(fp.get('exclusive_contributor'))}")
    lines.append(f"          · contributor_id:        {pct(fp.get('contributor_id'))}")
    lines.append(f"          · item_is_ai_generated:  {pct(fp.get('item_is_ai_generated'))}")
    lines.append(f"          · content_type:          {pct(fp.get('content_type'))}")
    lines.append(f"          · time_on_sales:         {pct(fp.get('time_on_sales'))}")
    lines.append(f"          · purchase:              {pct(fp.get('purchase'))}")

    lines.append("  • Exclusive contributors:")
    lines.append(f"      – Proportion of exclusive items: {pct(ex.get('proportion_items'))}")
    lines.append(f"      – Unique exclusive contributors: {num(ex.get('unique_contributors'))}")
    lines.append(f"      – Average ranks (exclusive only): first={num(ex.get('avg_first_rank'))}, last={num(ex.get('avg_last_rank'))}, mean={num(ex.get('avg_mean_rank'))}")
    tk = ex.get("topk_share", {}) or {}
    if tk:
        pairs = ", ".join([f"Top-{k}: {pct(v)}" for k, v in sorted(tk.items())])
        lines.append(f"      – Top-K exclusive share: {pairs}")

    lines.append("  • Freshness (days on sale from time_on_sales):")
    lines.append(f"      – Avg latest: {num(fr.get('avg_latest'))}  |  Avg oldest: {num(fr.get('avg_oldest'))}")
    lines.append(f"      – Mean: {num(fr.get('avg_mean'))}  |  Median(mean): {num(fr.get('median_mean'))}  |  Std(mean): {num(fr.get('std_mean'))}")
    lines.append(f"      – Cold-start share (≤7d): {pct(fr.get('coldstart_le_7_share'))}  |  (≤30d): {pct(fr.get('coldstart_le_30_share'))}")

    lines.append("  • Diversity:")
    lines.append(f"      – Contributors normalized entropy: {pct(dv.get('contributors_normalized_entropy'))}")
    lines.append(f"      – Content normalized entropy:      {pct(dv.get('content_normalized_entropy'))}")
    lines.append(f"      – Avg unique contributors per query: {num(dv.get('unique_contributors_avg'))}")
    lines.append(f"      – Avg unique content categories/query: {num(dv.get('unique_content_avg'))}")

    lines.append("  • Photo AI:")
    lines.append(f"      – AI rate: {pct(ph.get('ai_impression_rate'))}  |  Non-AI rate: {pct(ph.get('non_ai_impression_rate'))}  |  Unknown: {pct(ph.get('unknown_impression_rate'))}")

    lines.append("  • Sales (based on 'purchase'; lower rank is better):")
    lines.append(f"      – Avg purchase rank: {num(sa.get('avg_purchase_rank'))}  |  Median: {num(sa.get('median_purchase_rank'))}")
    topn_multi = sa.get("topn_shares", {}) or {}
    if topn_multi:
        parts = ", ".join([f"Top-{k}: {pct(v)}" for k, v in sorted(topn_multi.items())])
        lines.append(f"      – Top-N shares: {parts}")

    return "\n".join(lines)
