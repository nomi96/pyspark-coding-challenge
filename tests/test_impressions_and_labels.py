
import datetime as dt
from pyspark.sql import functions as F
from pipeline import build_training_inputs
from utils.schemas import impressions_schema, clicks_schema, add_to_carts_schema, previous_orders_schema


def test_impressions_array_preserved_and_slot_labels(spark):
    imps = [{
        "dt": "2025-01-10",
        "ranking_id": "rk",
        "customer_id": 9,
        "impressions": [{"item_id": 1, "is_order": False}, {"item_id": 2, "is_order": True}, {"item_id": 3, "is_order": False}],
    }]
    out = build_training_inputs(
        spark.createDataFrame(imps, schema=impressions_schema),
        spark.createDataFrame([], schema=clicks_schema),
        spark.createDataFrame([], schema=add_to_carts_schema),
        spark.createDataFrame([], schema=previous_orders_schema),
        history_length=3,
        lookback_days=365,
    )
    rows = out.orderBy("impression_index").collect()
    # same impressions list for all slots
    assert [r.impressions for r in rows] == [1,2,3]
    # item_id and is_order reflect slot
    assert rows[0].item_id == 1 and rows[0].is_order == False
    assert rows[1].item_id == 2 and rows[1].is_order == True
    assert rows[2].item_id == 3 and rows[2].is_order == False
