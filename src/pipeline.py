from pyspark.sql import functions as F, Window as W
from pyspark.sql import DataFrame
from utils.utils import dt_to_ts_start, clamp_array_pair

# Action type codes
CLICK_CODE = 1
ATC_CODE   = 2
ORD_CODE   = 3

def _normalize_actions(clicks_df: DataFrame, atc_df: DataFrame, orders_df: DataFrame) -> DataFrame:
    """
    Union all action sources into a single normalized table:
    columns: customer_id (long), item_id (long), action_time (timestamp), action_type (int)
    """

    CLICK = F.lit(CLICK_CODE).cast("int")
    ATC   = F.lit(ATC_CODE).cast("int")
    ORD   = F.lit(ORD_CODE).cast("int")

    # clicks
    clicks_norm = (
        clicks_df
        .select(
            "customer_id",
            F.col("item_id").cast("long").alias("item_id"),
            F.to_timestamp(F.col("click_time")).alias("action_time"),
            CLICK.alias("action_type"),
        )
    )

    # add-to-carts
    atc_norm = (
        atc_df
        .select(
            "customer_id",
            F.col("config_id").cast("long").alias("item_id"),   # config_id is the item id
            F.to_timestamp(F.col("occurred_at")).alias("action_time"),
            ATC.alias("action_type"),
        )
    )

    # previous_orders:
    # - If a richer timestamp 'occurred_at' exists, prefer it.
    # - Else, derive from order_date at midnight.
    has_occurred = "occurred_at" in orders_df.columns
    po_ts = (
        F.to_timestamp("occurred_at")
        if has_occurred
        else F.to_timestamp(F.col("order_date").cast("string"))
    )

    # orders (date -> timestamp at 00:00:00)
    orders_norm = (
        orders_df
        .select(
            "customer_id",
            F.col("config_id").cast("long").alias("item_id"),
            po_ts.alias("action_time"),
            #F.to_timestamp(F.col("order_date")).alias("action_time"),
            ORD.alias("action_type"),
        ).where(F.col("action_time").isNotNull())
    )

    # Co-locate by user to make later groupBy/joins cheaper
    
    return clicks_norm.unionByName(atc_norm).unionByName(orders_norm)

def build_training_inputs(
    impressions_df: DataFrame,
    clicks_df: DataFrame,
    atc_df: DataFrame,
    orders_df: DataFrame,
    history_length: int = 1000,
    lookback_days: int = 365,
) -> DataFrame:
    """
    Produce per-impression rows with columns:
      - dt (string 'YYYY-MM-DD')
      - ranking_id (string)
      - customer_id (long)
      - impression_index (int)
      - item_id (long)            # the current candidate in this slot
      - is_order (boolean)        # label indicating if the impression item was ordered that day
      - impressions (int)         # the item_id of impressions row
      - actions (array<long>)     # newest→oldest; 0-padded to `history_length`
      - action_types (array<int>) # aligned with actions; 1/2/3/0

    Performance notes:
    - Pure DataFrame APIs; no Python UDFs.
    - Range join on (customer_id) + date window; avoids broadcasting big tables.
    - repartitioning on customer_id to reduce shuffle skew.
    - Window + row_number to trim to history_length, then aggregation to arrays.
    - Column pruning everywhere.

    GPU guidance:
    - With RAPIDS Accelerator enabled (spark.rapids.sql.enabled=true), all the
      projections/filters/joins/window sorts are GPU-accelerated automatically,
      provided types are supported (which they are here).
    """

    actions_all = _normalize_actions(clicks_df, atc_df, orders_df).cache()
    
    # pre-partition actions by customer_id to reduce shuffle on the range join
    
    actions_all = actions_all.repartition(F.spark_partition_id(), "customer_id")

    # Prepare impressions: explode into one row per item
    base = (
        impressions_df
        .withColumn("dt_ts", dt_to_ts_start(F.col("dt")))
        .select(
            "dt",
            "dt_ts",
            "ranking_id",
            "customer_id",
            F.posexplode("impressions").alias("impression_index", "impression"),
        )
        .select(
            F.col("dt"),
            F.col("dt_ts"),
            F.col("ranking_id"),
            F.col("customer_id").cast("long").alias("customer_id"),
            F.col("impression_index").cast("int").alias("impression_index"),
            F.col("impression.item_id").cast("long").alias("item_id"),
            F.col("impression.is_order").alias("is_order"),
        )
    )
    
    # (Optional) pre-partition actions by customer_id to reduce shuffle on the range join
    
    base = base.repartition(F.spark_partition_id(), "customer_id")

    # Join actions to impressions by customer, with time filter via a condition
    b = base.alias("b")
    a = actions_all.alias("a")

    # Scoped actions (anti-leakage & lookback happen during join condition)
    # Lookback lower bound (same for all action types)
    
    lower_bound = F.expr(f"b.dt_ts - INTERVAL {lookback_days} DAYS")

    # Type-specific upper bounds:
    # - clicks/add-to-carts: strictly before impression day
    # - orders: strictly before (impression day - 1 day)
    cond = (
        (F.col("a.action_time") >= lower_bound) &
        (
            (
                F.col("a.action_type").isin(CLICK_CODE, ATC_CODE) &
                (F.col("a.action_time") < F.col("b.dt_ts"))
            ) |
            (
                (F.col("a.action_type") == F.lit(ORD_CODE)) &
                (F.col("a.action_time") < F.expr("b.dt_ts - INTERVAL 1 DAYS"))
            )
        )
    )

    #Join actions to impressions with explicit select of normalized columns
    joined = (
        b.join(a, on=((F.col("b.customer_id") == F.col("a.customer_id"))), how="left")
        .select(
            # keep all normalized impression columns
            F.col("b.dt").alias("dt"),
            F.col("b.dt_ts").alias("dt_ts"),
            F.col("b.ranking_id").alias("ranking_id"),
            F.col("b.customer_id").alias("customer_id"),
            F.col("b.impression_index").alias("impression_index"),
            F.col("b.item_id").alias("item_id"),
            F.col("b.is_order").alias("is_order"),
            # action columns (may be null)
            F.col("a.item_id").cast("long").alias("a_item_id"),
            F.col("a.action_type").cast("int").alias("a_action_type"),
            F.col("a.action_time").alias("a_action_time"),
        ).where(cond)
        .repartition("dt", "ranking_id", "customer_id", "impression_index")
    )

    # Rank actions per (customer, dt, ranking_id, impression_index) by recency
    win = (
        W.partitionBy("dt", "ranking_id", "customer_id", "impression_index") 
         .orderBy(F.col("a_action_time").desc())
    )

    ranked = (
        joined
        .withColumn("rn", F.when(F.col("a_action_time").isNotNull(), F.row_number().over(win)))
        .where((F.col("rn").isNull()) | (F.col("rn") <= history_length))
    )

    # Aggregate to arrays, ordered by recency
    # 5) Deterministic ordering inside the aggregate:
    #    collect structs (time, value, type) -> sort desc by time -> project arrays
    aggregated = (
        ranked
        .groupBy("dt", "dt_ts", "ranking_id", "customer_id", "impression_index", "item_id", "is_order")
        .agg(
            F.sort_array(
                F.collect_list(
                    F.struct(
                        F.col("a_action_time").alias("ts"),
                        F.coalesce(F.col("a_item_id"), F.lit(0).cast("long")).alias("v"),
                        F.coalesce(F.col("a_action_type"), F.lit(0).cast("int")).alias("t"),
                    )
                ),
                asc=False,
            ).alias("hist")
        )
    )

    # For impressions with no actions at all, we still need rows; fill missing aggregates with empty arrays
    #Left-join back to base to keep rows with *no* actions and coalesce
    agg = aggregated.alias("agg")
    keep_keys = ["dt", "dt_ts", "ranking_id", "customer_id", "impression_index", "item_id", "is_order"]

    joined2 = (
        b.join(
            agg,
            on=[F.col(f"b.{k}") == F.col(f"agg.{k}") for k in keep_keys],
            how="left",
        )
        .select(
            *[F.col(f"b.{k}").alias(k) for k in keep_keys],
            F.coalesce(F.col("agg.hist"), F.array().cast("array<struct<ts:timestamp,v:bigint,t:int>>")).alias("hist"),
        )
    )

    # Pad/trim to exact history_length (keep order: most recent first)
    #Project to arrays and clamp/pad to fixed length
    with_arrays = joined2.select(
        *[F.col(k) for k in keep_keys],
        F.transform(F.col("hist"), lambda x: x["v"]).alias("actions_raw"),
        F.transform(F.col("hist"), lambda x: x["t"]).alias("action_types_raw"),
    )

    actions_fixed, action_types_fixed = clamp_array_pair("actions_raw", "action_types_raw", history_length)

    out = (
        with_arrays
        .withColumn("impressions", F.col("item_id").cast("long"))
        .withColumn("actions", actions_fixed)
        .withColumn("action_types", action_types_fixed)
        .drop("actions_raw", "action_types_raw", "hist", "dt_ts")
        .orderBy("dt", "ranking_id", "customer_id", "impression_index")
    )

    return out