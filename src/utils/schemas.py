from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, IntegerType, ArrayType, BooleanType, TimestampType, DateType
)

# Impressions schema
impressions_schema = StructType([
    StructField("dt", StringType(), nullable=False),  # 'YYYY-MM-DD'
    StructField("ranking_id", StringType(), nullable=False),
    StructField("customer_id", LongType(), nullable=False),
    StructField("impressions", ArrayType(
        StructType([
            StructField("item_id", LongType(), nullable=False),
            StructField("is_order", BooleanType(), nullable=False),
        ])
    ), nullable=False),
])

# Clicks schema
clicks_schema = StructType([
    StructField("dt", StringType(), nullable=False),  # 'YYYY-MM-DD'
    StructField("customer_id", LongType(), nullable=False),
    StructField("item_id", LongType(), nullable=False),
    StructField("click_time", StringType(), nullable=False),
])

# Add to carts schema
add_to_carts_schema = StructType([
    StructField("dt", StringType(), nullable=False),  # 'YYYY-MM-DD'
    StructField("customer_id", LongType(), nullable=False),
    StructField("config_id", LongType(), nullable=False),  # same as item_id
    StructField("simple_id", LongType(), nullable=True),
    StructField("occurred_at", StringType(), nullable=False),
])

# Previous orders schema
previous_orders_schema = StructType([
    StructField("order_date", DateType(), nullable=False),
    StructField("customer_id", LongType(), nullable=False),
    StructField("config_id", LongType(), nullable=False),  # same as item_id
    StructField("simple_id", IntegerType(), nullable=True),       # added
    StructField("occurred_at", TimestampType(), nullable=False),
])