import datetime as dt
from pyspark.sql import functions as F
from pipeline import build_training_inputs
from utils.schemas import impressions_schema, clicks_schema, add_to_carts_schema, previous_orders_schema

def test_history_padding_and_ordering(spark):
    # one customer, many actions across days
    impressions = [{
        "dt": "2025-01-10",
        "ranking_id": "r1",
        "customer_id": 1,
        "impressions": [{"item_id": 111, "is_order": False}, {"item_id": 222, "is_order": True}],
    }]

    # clicks on previous days + same day (same-day should be excluded)
    clicks = [
        {"dt": "2025-01-09", "customer_id": 1, "item_id": 111, "click_time": "2025-01-09 13:00:00"},
        {"dt": "2025-01-08", "customer_id": 1, "item_id": 222, "click_time": "2025-01-08 13:00:00"},
        {"dt": "2025-01-10", "customer_id": 1, "item_id": 30, "click_time": "2025-01-10 01:00:00"},  # exclude
    ]
    clicks_df = spark.createDataFrame(clicks, schema=clicks_schema)

    atc = [
        {"dt": "2025-01-07", "customer_id": 1, "config_id": 40, "simple_id": 1, "occurred_at": "2025-01-07 12:00:00"},
    ]
    atc_df = spark.createDataFrame(atc, schema=add_to_carts_schema)

    orders = [
        {"order_date": dt.date(2024, 12, 31), "customer_id": 1, "config_id": 50,"simple_id": 10, "occurred_at": dt.datetime(2024, 12, 31, 0, 0)},
    ]
    orders_df = spark.createDataFrame(orders, schema=previous_orders_schema)

    imps_df = spark.createDataFrame(impressions, schema=impressions_schema)

    out = build_training_inputs(imps_df, clicks_df, atc_df, orders_df, history_length=5, lookback_days=400)

    row = out.collect()[0]
    assert row.customer_id == 1
    # Most recent first: click(2025-01-09)=10, click(2025-01-08)=20, atc(2025-01-07)=40, order(2024-12-31)=50, then padding 0
    assert row.actions[:5] == [111, 222, 40, 50, 0]
    assert row.action_types[:5] == [1, 1, 2, 3, 0]

def test_no_actions_edge_case(spark):
    impressions = [{
        "dt": "2025-01-10",
        "ranking_id": "r2",
        "customer_id": 2,
        "impressions": [{"item_id": 333, "is_order": False}],
    }]

    imps_df = spark.createDataFrame(impressions, schema=impressions_schema)
    empty = spark.createDataFrame([], schema=clicks_schema)
    atc_empty = spark.createDataFrame([], schema=add_to_carts_schema)
    orders_empty = spark.createDataFrame([], schema=previous_orders_schema)

    out = build_training_inputs(imps_df, empty, atc_empty, orders_empty, history_length=4, lookback_days=365)

    row = out.collect()[0]
    assert row.actions == [0, 0, 0, 0]
    assert row.action_types == [0, 0, 0, 0]

def test_leakage_exclusion_same_day(spark):
    impressions = [{
        "dt": "2025-01-10",
        "ranking_id": "r3",
        "customer_id": 3,
        "impressions": [{"item_id": 999, "is_order": False}],
    }]

    clicks = [
        {"dt": "2025-01-10", "customer_id": 3, "item_id": 77, "click_time": "2025-01-10 00:00:00"},  # exclude (same day)
        {"dt": "2025-01-09", "customer_id": 3, "item_id": 88, "click_time": "2025-01-09 23:59:59"},
    ]

    out = build_training_inputs(
        spark.createDataFrame(impressions, schema=impressions_schema),
        spark.createDataFrame(clicks, schema=clicks_schema),
        spark.createDataFrame([], schema=add_to_carts_schema),
        spark.createDataFrame([], schema=previous_orders_schema),
        history_length=3,
        lookback_days=30,
    )

    row = out.collect()[0]
    assert row.actions == [88, 0, 0]
    assert row.action_types == [1, 0, 0]