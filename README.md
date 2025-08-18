
# pyspark-coding-challenge

This project builds **per‑impression** training inputs for a transformer
model. The pipeline consumes four event streams (impressions, clicks, add‑to‑carts,
previous orders) and emits fixed‑shape tensors per impression so model training can
run efficiently and reproducibly at scale.

---

## 1) High‑Level Proposed Structure for Training Inputs

Each **row** in the final dataset corresponds to a **single impression slot**. For a batch of `N` exploded impressions,
we produce the following tensors and context features:

### Tensors (for the model)
- **`impressions`** — shape **[N]**  
  Integer embedding index for the item in the slot (equal to `item_id`).

- **`actions`** — shape **[N, 1000]**  
  The **most recent 1000** historical customer actions **strictly before** `dt`
  (recency‑sorted descending: most recent first). Values are integer embedding indices:
  - click → `item_id`
  - add‑to‑cart → `config_id` (config‑level embedding)
  - previous order → `config_id`  
  `0` is used for padding.

- **`action_types`** — shape **[N, 1000]**  
  Mirrors `actions`; type encoding: `1=click`, `2=add_to_cart`, `3=order`, `0=pad`.

### Context/Labels (for bookkeeping & analysis)
```
dt (string), ranking_id (string), customer_id (long),
impression_index (int), item_id (long), is_order (boolean),
impressions (long), actions (array<long>[1000]), action_types (array<int>[1000])
```

> Why this structure?  
> *Per‑impression rows* let us pair a target (did this slot convert? which item was shown?)
> with a **fixed‑size** representation of the user’s history for efficient batching on GPU/TPU.
> Keeping clicks/ATC/orders in one sequence with a `type` channel preserves temporal
> structure while remaining simple to embed and feed to sequence models (Transformers, RNNs,
> or temporal Conformers).
---

## 2) High‑Level Description of How Training Will Work

1. **Input Layers / Embeddings**
   - Look up **item embeddings** for `impressions` (the candidate item in the slot).
   - Look up **event embeddings** for `actions` (shared embedding table keyed by integer id).
   - Optionally add a **type embedding** for `action_types` and sum with event embeddings.
   - (Optional) Add **position encodings** over the 1000‑length history window.

2. **User History Encoder**
   - Encode the `[1000 × d]` action sequence via a Transformer/Temporal CNN (causal or
     bidirectional, depending on leakage tolerance). Mask padding where value==0.
   - Pool (e.g., attention pooling, mean, CLS token) to produce a **user context vector**.

3. **Scoring Head**
   - Concatenate the user context with the **candidate item embedding** (from `impressions`).
   - Pass through an MLP to produce a **score** for the impression (click‑through,
     add‑to‑cart, purchase likelihood, or a multi‑task head if desired).

4. **Loss / Targets**
   - Use `is_order` as a supervised label for **order‑intent** if the task is purchase,
     or curate alternative labels (e.g., post‑impression click) from logs.
   - Optimize with cross‑entropy / BCE / focal loss depending on class balance.

5. **Negative Sampling (optional)**
   - Train with **in‑batch negatives**: other impressions within the same batch serve as
     non‑clicked/non‑purchased negatives for contrastive or cross‑entropy setups.

6. **Serving Alignment**
   - The same history encoder can serve cached **user contexts** online; scoring combines
     these with candidate item embeddings in real time.

---

## 3) Why These Pipelines?

- **Leakage‑safe**: Only actions strictly **before midnight of `dt`** are included.
- **Deterministic & Reproducible**: Fixed length (1000) with defined ordering, padding,
  and truncation rules makes training deterministic across runs.
- **Simple Join Keys**: The pipeline keys everything by `customer_id` and time, which
  scales naturally in distributed settings and simplifies backfills.
- **Model‑agnostic**: Works for classic sequence models, shallow MLPs over pooled histories,
  and modern retrieval‑based architectures.

---

## 4) The Pipelines We Built

### 4.1 Input Normalization
- **Impressions**: explode `impressions` array into rows → (`dt`, `ranking_id`, `customer_id`,
  `impression_index`, `item_id`, `is_order`); set `impressions=item_id` (scalar).
- **Clicks**: normalize to (`customer_id`, `item_id`, `action_ime`, `type=1`).
- **Add_To_Cart**: normalize to (`customer_id`, `item_id=config_id`, `action_ime`, `type=2`).  
- **Previous_Orders**: normalize to (`customer_id`, `item_id=config_id`, `action_time=order_date 00:00:00`, `type=3`).

### 4.2 Filtering & Windowing
- Compute `dt_ts = to_timestamp(dt, 'YYYY-MM-DD')` (midnight of the impression day).
- Keep events with `action_time < dt_ts` (no same‑day leakage).
- Apply `lookback_days` via `action_time >= dt_ts - interval lookback_days days`.

### 4.3 History Construction (Top‑K 1000, Recency)
- Union normalized events; sort by `action_time DESC` **per customer**.
- Take top **1000** events per customer.
- Left‑join back onto exploded impressions by `(customer_id, dt, ranking_id)`;
  repeat the user‑level history across all slots of that ranking **without
  re‑shuffling** (slots remain independent via their own `impressions` scalar).
- **Pad** to 1000 with zeros and emit `actions` and `action_types` arrays.

---

## 5) Performance Considerations & How the PySpark is Written for Scale

The implementation aims to be **shuffle‑aware**, **memory‑efficient**, and **vectorized**:

1. **Columnar Ops & Spark SQL Functions**
   - Use built‑in expressions (`to_timestamp`, `struct`, `sort_array`, `slice`, `transform`,
     `sequence`, etc.) to stay in JVM/WholeStageCodegen paths.
   - Avoid UDFs/UDTFs in critical loops to reduce Python‑JVM crossings.

2. **Early Projections & Predicate Pushdown**
   - Select only necessary columns as early as possible.
   - Apply `action_time` window filters **before** any expensive join/groupBy to reduce shuffle size.

3. **Skew & Partitioning**
   - Repartition by **`customer_id`** when unioning actions to keep per‑customer operations local.
   - For large backfills, also partition/bucket inputs by `dt` to parallelize by time slices.
   - Consider salting very hot customers if extreme skew is observed.

4. **Top‑K without Full Sort**
   - Use `sortWithinPartitions` followed by a **per‑key limit** (Window with `row_number` or
     `aggregate` over `collect_list` + `slice`) to avoid global sorts.
   - Keep **K=1000** small enough to fit in memory during aggregation; arrays are bounded.

5. **Broadcast Joins Where Appropriate**
   - If impression sets per `dt` are small relative to actions, broadcast them;
     otherwise, broadcast the **top‑K histories** (post‑reduction) to minimize network IO.

6. **Caching Checkpoints (Optional)**
   - Persist the normalized‑and‑filtered actions dataset if reused across multiple
     training slices or ablations.
   - Use `storageLevel=MEMORY_AND_DISK` to tolerate data skew without OOM.

7. **Serialization & Output**
   - Write **Parquet** with compression (e.g., `snappy`) and a sensible `spark.sql.files.maxPartitionBytes`
     to balance parallelism vs small‑file overhead.
   - Use `mode="overwrite"` in CI/testing; `append` with partitioning by `dt` in production.

> The code includes comments on where repartitioning, broadcasting, or caching are helpful.
> These settings are workload‑dependent; use the Spark UI to confirm stages/shuffles at scale.
---

## 6) Repository Layout

```
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── pytest.ini
├── src
    ├── __init__.py
    ├── pipeline.py
    └── utils
    │   ├── schemas.py
    │   └── utils.py
└── tests
    ├── conftest.py
    ├── test_edge_cases.py
    ├── test_history_build.py
    ├── test_history_logic.py
    ├── test_impressions_and_labels.py
    ├── test_lookback_and_limits.py
    ├── test_pipeline_contract.py
    └── test_pipeline_end_to_end.py         
```

---

## 7) How to run locally

### Prereqs
- Python 3.9+
- PySpark 3.3+ (or any Spark 3.x)
- pytest

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### Running the pipeline (example)
Below we show a **pure-DataFrame** run using small in-memory data (so you can test quickly):

```python
from pyspark.sql import SparkSession
from pipeline import build_training_inputs
from utils.schemas import impressions_schema, clicks_schema, add_to_carts_schema, previous_orders_schema

spark = SparkSession.builder.master("local[2]").appName("pyspark_coding_challenge").getOrCreate()

# Create inputs (replace with your own DataFrames or reads)
impressions_df = spark.createDataFrame([...], schema=impressions_schema)
clicks_df      = spark.createDataFrame([...], schema=clicks_schema)
atc_df         = spark.createDataFrame([...], schema=add_to_carts_schema)
orders_df      = spark.createDataFrame([...], schema=previous_orders_schema)

out_df = build_training_inputs(
    impressions_df,
    clicks_df,
    atc_df,
    orders_df,
    history_length=1000,
    lookback_days=365,
)

# Show / write
out_df.show(truncate=False)
out_df.write.mode("overwrite").partitionBy("dt").parquet("output/training_inputs")
```

You can also **replace** the in-memory inputs with your own readers (`spark.read.table`, `spark.read.parquet`, etc).
The pipeline only depends on the DataFrame columns, not the storage layouts.

---

## 8) Test & QA

Run all tests locally:

```bash
pytest -q
```

The suite covers:
- Anti-leakage (excludes same-day actions)
- Lookback windowing
- Exact 1000-length arrays with correct **ordering** and **padding**
- Correct **action_type** mapping (1, 2, 3; with 0-padding)
- Customers with **no actions** / **few actions** / **many actions**
- Mixed sources & duplicate items (kept by design)
- Stability across multiple impression days

## 9) Notes & assumptions

- `previous_orders.order_date` is a **date** (no timestamp). We coerce to `00:00:00` of that day; since we exclude **same-day** actions, orders on `dt` are excluded.
- `add_to_carts.config_id` is **the same field as item_id**; `simple_id` is ignored (not needed by the model).
- We **retain duplicates** in the action history. This is explicitly useful information per the spec.
- If you need different history length (e.g., 512/2048), pass `history_length=<N>`.
- If you want a longer lookback than one year, set `lookback_days`.

## 10) Why this pipeline

It meets the spec precisely (no leakage, correct typing, 1000-length history) while being:
- **Simple**: few, well-commented transformations.
- **Scalable**: native Spark ops that push down and avoid UDFs.
- **Composable**: pure-DataFrame API so you can plug in your storage and schedulers.
- **Trainer-friendly**: exploded, partitioned outputs for fast sampling and stable, fixed-shape tensors.
