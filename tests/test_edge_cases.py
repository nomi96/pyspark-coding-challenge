
import datetime as dt
from pyspark.sql import functions as F
from pipeline import build_training_inputs
from utils.schemas import impressions_schema, clicks_schema, add_to_carts_schema, previous_orders_schema


def test_empty_actions_tables(spark):
    imps = [{
        "dt": "2025-01-10",
        "ranking_id": "r0",
        "customer_id": 42,
        "impressions": [{"item_id": 1001, "is_order": False}],
    }]
    out = build_training_inputs(
        spark.createDataFrame(imps, schema=impressions_schema),
        spark.createDataFrame([], schema=clicks_schema),
        spark.createDataFrame([], schema=add_to_carts_schema),
        spark.createDataFrame([], schema=previous_orders_schema),
        history_length=5,
        lookback_days=365,
    )
    r = out.first()
    assert r.actions == [0,0,0,0,0]
    assert r.action_types == [0,0,0,0,0]
    assert r.impressions == 1001

def test_duplicates_retained(spark):
    imps = [{"dt": "2025-01-10", "ranking_id": "rd", "customer_id": 7, "impressions": [{"item_id": 5, "is_order": False}]}]
    clicks = [
        {"dt": "2025-01-09", "customer_id": 7, "item_id": 5, "click_time": "2025-01-09 09:00:00"},
        {"dt": "2025-01-08", "customer_id": 7, "item_id": 5, "click_time": "2025-01-08 08:00:00"},
    ]
    out = build_training_inputs(
        spark.createDataFrame(imps, schema=impressions_schema),
        spark.createDataFrame(clicks, schema=clicks_schema),
        spark.createDataFrame([], schema=add_to_carts_schema),
        spark.createDataFrame([], schema=previous_orders_schema),
        history_length=4,
        lookback_days=365,
    )
    r = out.first()
    # Both duplicate item_ids retained and ordered by recency
    assert r.actions[:2] == [5,5]
    assert r.action_types[:2] == [1,1]

def test_multiple_customers_and_rankings(spark):
    imps = [
        {"dt": "2025-01-10", "ranking_id": "ra", "customer_id": 1, "impressions": [{"item_id": 101, "is_order": False}]},
        {"dt": "2025-01-10", "ranking_id": "rb", "customer_id": 2, "impressions": [{"item_id": 201, "is_order": True}]},
    ]
    clicks = [
        {"dt": "2025-01-09", "customer_id": 1, "item_id": 101, "click_time": "2025-01-09 10:00:00"},
        {"dt": "2025-01-09", "customer_id": 2, "item_id": 201, "click_time": "2025-01-09 10:00:00"},
    ]
    out = build_training_inputs(
        spark.createDataFrame(imps, schema=impressions_schema),
        spark.createDataFrame(clicks, schema=clicks_schema),
        spark.createDataFrame([], schema=add_to_carts_schema),
        spark.createDataFrame([], schema=previous_orders_schema),
        history_length=3,
        lookback_days=365,
    )
    rows = {(r.ranking_id, r.customer_id): r for r in out.collect()}
    assert rows[("ra",1)].actions[0] == 101
    assert rows[("rb",2)].actions[0] == 201
