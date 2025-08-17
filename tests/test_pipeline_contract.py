import datetime as dt
from pyspark.sql import functions as F
from pipeline import build_training_inputs
from utils.schemas import impressions_schema, clicks_schema, add_to_carts_schema, previous_orders_schema
from pyspark.sql.types import StringType, BooleanType, ArrayType, LongType, IntegerType

def test_schema_contract_and_shapes(spark):
    imps = [{
        "dt": "2025-01-10",
        "ranking_id": "r",
        "customer_id": 1,
        "impressions": [{"item_id": 11, "is_order": False}, {"item_id": 22, "is_order": True}],
    }]
    clicks = [{"dt": "2025-01-09", "customer_id": 1, "item_id": 99, "click_time": "2025-01-09 01:00:00"}]
    atc    = [{"dt": "2025-01-08", "customer_id": 1, "config_id": 55, "simple_id": 1, "occurred_at": "2025-01-08 02:00:00"}]
    # the DateType above is awkward to craft inline; construct via DataFrame:
    orders_df = spark.createDataFrame([("2025-01-01", 1, 77)], "order_date string, customer_id int, config_id int").withColumn("order_date", F.to_date("order_date"))
    imps_df = spark.createDataFrame(imps, schema=impressions_schema)
    clicks_df = spark.createDataFrame(clicks, schema=clicks_schema)
    atc_df = spark.createDataFrame(atc, schema=add_to_carts_schema)

    out = build_training_inputs(imps_df, clicks_df, atc_df, orders_df, history_length=4, lookback_days=365)

    # Required columns
    expected_cols = {"dt","ranking_id","customer_id","impression_index","item_id","is_order","impressions","actions","action_types"}
    assert expected_cols.issubset(set(out.columns))

    # dtypes
    schema = {f.name: f.dataType for f in out.schema.fields}
    assert isinstance(schema["dt"], StringType)
    assert isinstance(schema["ranking_id"], StringType)
    assert isinstance(schema["customer_id"], LongType)      # cast to long
    assert isinstance(schema["impression_index"], IntegerType)
    assert isinstance(schema["is_order"], BooleanType)
    assert isinstance(schema["impressions"], LongType)
    assert isinstance(schema["actions"], ArrayType)
    assert isinstance(schema["action_types"], ArrayType)

    # fixed-length arrays
    row0 = out.orderBy("impression_index").first()
    assert len(row0.actions) == 4
    assert len(row0.action_types) == 4
