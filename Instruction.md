# flow 
```
[Dataset (events.jsonl, metadata...)]
        │
        ├─(1) Import offline  →  Tính features & lưu state
        │
        ├─(2) Train offline   →  Huấn luyện & LƯU MODEL vào store
        │
        └─(3) Serve online    →  API đọc model đã lưu để rerank
                                  (không train trong lúc serve)
```

# run redis
docker run -d --name redis -p 6379:6379 redis:7

# import 
docker run --rm -v "$PWD:/work" -w /work metarank/metarank:latest \
  import --config /work/config.yml --data /work/data/events.jsonl

# train
docker run --rm -v "$PWD:/work" -w /work metarank/metarank:latest \
  train  --config /work/config.yml

# serve
docker run --rm --name metarank-serve -p 9090:8080 \
  -v "$PWD:/work" -w /work metarank/metarank:latest \
  serve --config /work/config.yml


# smoke-test
python - <<'PY'
import json, time, requests
ids = list(json.load(open('data/metadata.json')).keys())[:5]
payload = {"id":"rq_smoke","timestamp":int(time.time()*1000),
           "user":"u_eval","session":"s_eval","items":[{"id":i} for i in ids]}
r = requests.post("http://localhost:9090/rank/xgboost", json=payload, timeout=30)
print("STATUS:", r.status_code)
print("BODY:", r.text[:500])
PY


# run online metrics 
python3 run_online_metrics.py --host http://localhost:9090 --model xgboost