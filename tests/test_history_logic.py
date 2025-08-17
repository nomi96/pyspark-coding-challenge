
import datetime as dt
from pyspark.sql import functions as F
from pipeline import build_training_inputs
from utils.schemas import impressions_schema, clicks_schema, add_to_carts_schema, previous_orders_schema


def test_same_day_actions_are_excluded(spark):
    imps = [{
        "dt": "2025-01-10",
        "ranking_id": "r",
        "customer_id": 1,
        "impressions": [{"item_id": 1, "is_order": False}],
    }]
    # All three sources include same-day and previous-day entries
    clicks = [
        {"dt": "2025-01-10", "customer_id": 1, "item_id": 10, "click_time": "2025-01-10 07:00:00"},  # same-day (exclude)
        {"dt": "2025-01-09", "customer_id": 1, "item_id": 11, "click_time": "2025-01-09 07:00:00"},  # prev day (keep)
    ]
    atc = [
        {"dt": "2025-01-10", "customer_id": 1, "config_id": 20, "simple_id": 2, "occurred_at": "2025-01-10 08:00:00"},  # same-day (exclude)
        {"dt": "2025-01-08", "customer_id": 1, "config_id": 21, "simple_id": 2, "occurred_at": "2025-01-08 08:00:00"},  # keep
    ]
    # order_date is DateType -> use DataFrame to build proper type
    orders_df = (spark.createDataFrame([("2025-01-10", 1, 30)], "order_date string, customer_id int, config_id int")  # same-day (exclude)
                 .union(spark.createDataFrame([("2025-01-01", 1, 31)], "order_date string, customer_id int, config_id int"))  # keep
                 .withColumn("order_date", F.to_date("order_date")))

    imps_df = spark.createDataFrame(imps, schema=impressions_schema)
    clicks_df = spark.createDataFrame(clicks, schema=clicks_schema)
    atc_df = spark.createDataFrame(atc, schema=add_to_carts_schema)

    out = build_training_inputs(imps_df, clicks_df, atc_df, orders_df, history_length=6, lookback_days=365)
    r = out.first()
    # ensure same-day ids (10,20,30) are not present
    assert 10 not in r.actions and 20 not in r.actions and 30 not in r.actions
    # ensure previous-day/earlier ids are present
    assert 11 in r.actions and 21 in r.actions and 31 in r.actions

def test_ordering_desc_by_time_and_padding(spark):
    imps = [{"dt": "2025-02-01", "ranking_id": "r2", "customer_id": 1, "impressions": [{"item_id": 1, "is_order": False}]}]
    clicks = [
        {"dt": "2025-01-31", "customer_id": 1, "item_id": 101, "click_time": "2025-01-31 10:00:00"},
        {"dt": "2025-01-30", "customer_id": 1, "item_id": 102, "click_time": "2025-01-30 11:00:00"},
    ]
    atc = [{"dt": "2025-01-29", "customer_id": 1, "config_id": 201, "simple_id": 1, "occurred_at": "2025-01-29 12:00:00"}]
    orders_df = spark.createDataFrame([("2025-01-01", 1, 301)], "order_date string, customer_id int, config_id int").withColumn("order_date", F.to_date("order_date"))

    out = build_training_inputs(
        spark.createDataFrame(imps, schema=impressions_schema),
        spark.createDataFrame(clicks, schema=clicks_schema),
        spark.createDataFrame(atc, schema=add_to_carts_schema),
        orders_df,
        history_length=6,
        lookback_days=365,
    )
    r = out.first()
    print(r.impressions,r.actions, r.action_types)
    # Expect newest first: 101 (Jan31 10:00), 102 (Jan30 11:00), 201 (Jan29 12:00), 301 (Jan01 midnight)
    assert r.actions[:4] == [101, 102, 201, 301]
    assert r.action_types[:4] == [1, 1, 2, 3]
    # padded to length
    assert len(r.actions) == 6 and len(r.action_types) == 6
