
import datetime as dt
from pyspark.sql import functions as F
from pipeline import build_training_inputs
from utils.schemas import impressions_schema, clicks_schema, add_to_carts_schema, previous_orders_schema


def test_lookback_window_excludes_older_actions(spark):
    imps = [{"dt": "2025-02-01", "ranking_id": "r", "customer_id": 1, "impressions": [{"item_id": 1, "is_order": False}]}]
    clicks = [
        {"dt": "2025-01-31", "customer_id": 1, "item_id": 900, "click_time": "2025-01-31 01:00:00"},  # within 10 days
        {"dt": "2025-01-10", "customer_id": 1, "item_id": 901, "click_time": "2025-01-10 01:00:00"},  # > 10 days (exclude)
    ]
    atc = []
    orders_df = spark.createDataFrame([("2025-01-25", 1, 800)], "order_date string, customer_id int, config_id int").withColumn("order_date", F.to_date("order_date"))

    out = build_training_inputs(
        spark.createDataFrame(imps, schema=impressions_schema),
        spark.createDataFrame(clicks, schema=clicks_schema),
        spark.createDataFrame(atc, schema=add_to_carts_schema),
        orders_df,
        history_length=5,
        lookback_days=10,
    )
    r = out.first()
    assert 900 in r.actions  # in window
    assert 901 not in r.actions  # outside window
    assert 800 in r.actions  # in window

def test_history_length_truncation(spark):
    imps = [{"dt": "2025-01-10", "ranking_id": "r", "customer_id": 1, "impressions": [{"item_id": 1, "is_order": False}]}]
    # 6 clicks in descending time order
    clicks = [{"dt": "2025-01-0{}".format(d), "customer_id": 1, "item_id": 100+d, "click_time": "2025-01-0{} 10:00:00".format(d)} for d in range(4,10)]
    out = build_training_inputs(
        spark.createDataFrame(imps, schema=impressions_schema),
        spark.createDataFrame(clicks, schema=clicks_schema),
        spark.createDataFrame([], schema=add_to_carts_schema),
        spark.createDataFrame([], schema=previous_orders_schema),
        history_length=4,
        lookback_days=365,
    )
    r = out.first()
    assert len(r.actions) == 4 and len(r.action_types) == 4
    # newest first: days 09,08,07,06
    assert r.actions == [109, 108, 107, 106]
    assert r.action_types == [1, 1, 1, 1]
