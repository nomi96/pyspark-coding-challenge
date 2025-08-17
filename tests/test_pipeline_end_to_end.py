import datetime as dt
from pipeline import build_training_inputs
from utils.schemas import impressions_schema, clicks_schema, add_to_carts_schema, previous_orders_schema

def test_end_to_end_multiple_impressions(spark):
    # Two customers, two days of impressions, mixed actions
    impressions = [
        {"dt": "2025-01-10", "ranking_id": "ra", "customer_id": 1,
         "impressions": [{"item_id": 1, "is_order": False}, {"item_id": 2, "is_order": True}]},
        {"dt": "2025-01-11", "ranking_id": "rb", "customer_id": 2,
         "impressions": [{"item_id": 3, "is_order": False}]},
    ]

    clicks = [
        {"dt": "2025-01-09", "customer_id": 1, "item_id": 101, "click_time": "2025-01-09 10:00:00"},
        {"dt": "2025-01-10", "customer_id": 2, "item_id": 201, "click_time": "2025-01-10 10:00:00"},
        {"dt": "2025-01-11", "customer_id": 2, "item_id": 202, "click_time": "2025-01-11 00:00:01"}, # exclude for 2025-01-11
    ]

    atc = [
        {"dt": "2025-01-08", "customer_id": 1, "config_id": 301, "simple_id": 10, "occurred_at": "2025-01-08 09:00:00"},
        {"dt": "2025-01-10", "customer_id": 2, "config_id": 302, "simple_id": 11, "occurred_at": "2025-01-10 09:00:00"},
    ]

    orders = [
        {"order_date": dt.date(2025, 1, 1), "customer_id": 1, "config_id": 401,"simple_id": 100, "occurred_at": dt.datetime(2025, 1, 1, 0, 0)},
        {"order_date": dt.date(2025, 1, 10), "customer_id": 2, "config_id": 402,"simple_id": 10, "occurred_at": dt.datetime(2025, 1, 10, 0, 0)},  # exclude for dt=2025-01-11
    ]

    imps_df = spark.createDataFrame(impressions, schema=impressions_schema)
    clicks_df = spark.createDataFrame(clicks, schema=clicks_schema)
    atc_df = spark.createDataFrame(atc, schema=add_to_carts_schema)
    orders_df = spark.createDataFrame(orders, schema=previous_orders_schema)

    out = build_training_inputs(imps_df, clicks_df, atc_df, orders_df, history_length=4, lookback_days=365)

    rows = { (r.dt, r.ranking_id, r.customer_id, r.impression_index): r for r in out.collect() }

    # Customer 1, dt=2025-01-10 should see actions from 2025-01-09 (click=101), 2025-01-08 (atc=301), 2025-01-01 (order=401)
    r = rows[("2025-01-10", "ra", 1, 0)]
    assert r.actions[:4] == [101, 301, 401, 0]
    assert r.action_types[:4] == [1, 2, 3, 0]

    # Customer 2, dt=2025-01-11 should see 2025-01-10 click=201 and atc=302, but exclude 2025-01-11 and order on 2025-01-10
    r2 = rows[("2025-01-11", "rb", 2, 0)]
    assert r2.actions[:4] == [201, 302, 0, 0]
    assert r2.action_types[:4] == [1, 2, 0, 0]