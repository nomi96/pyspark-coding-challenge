from pyspark.sql import functions as F

def dt_to_ts_start(col_dt):
    """
    Convert a `YYYY-MM-DD` string column to a start-of-day timestamp.
    """
    return F.to_timestamp(F.concat_ws(" ", col_dt, F.lit("00:00:00")))


def clamp_array_pair(values_col, types_col, length):
    """
    Clamp a pair of arrays (values, types) to exact same `length`, padding with 0.
    """
    v_slice = F.expr(f"slice({values_col}, 1, {length})")
    t_slice = F.expr(f"slice({types_col}, 1, {length})")
    v_pad_len = F.greatest(F.lit(0), F.lit(length) - F.size(v_slice))
    t_pad_len = F.greatest(F.lit(0), F.lit(length) - F.size(t_slice))
    v_padded = F.concat(v_slice, F.array_repeat(F.lit(0), v_pad_len))
    t_padded = F.concat(t_slice, F.array_repeat(F.lit(0), t_pad_len))
    return v_padded, t_padded