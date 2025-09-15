# Step by Step 

## 0. Setting 
``` python -m venv venv && source venv/bin/activate ```
``` pip install pandas pyarrow ```

- Convert data to metarank's format
``` python convert_feather_to_metarank.py ```

## 1. Start Redis + API Serve 
``` 
docker compose up -d redis metarank-serve
```

## 2. Import & Train (Use metarank-train)
- Import data 
```
docker compose run --rm metarank-train \
  import --config /work/config.yml --data /work/data/events.jsonl
```

- Trainning model & push to redis 
```
docker compose run --rm metarank-train \
  train --config /work/config.yml
```

## 3. Restart metarank-serve (to load new model)
```
docker compose restart metarank-serve
```

## 4. Run Evaluation 
```
python run_online_metrics.py \
  --host http://localhost:9090 \
  --model xgboost
```