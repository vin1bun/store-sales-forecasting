from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, mean, dayofweek, month, year
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("StoreSalesForecasting").getOrCreate()

df = spark.read.csv("master.csv", header=True, inferSchema=True)

df = df.withColumn("date", col("date").cast("date"))

window = Window.partitionBy("store_nbr", "family").orderBy("date")

df = df.withColumn("lag_7", lag("sales", 7).over(window))
df = df.withColumn("lag_14", lag("sales", 14).over(window))
df = df.withColumn("lag_28", lag("sales", 28).over(window))

rolling_window = window.rowsBetween(-6, 0)
df = df.withColumn("rolling_mean_7", mean("sales").over(rolling_window))

df = df.withColumn("dayofweek", dayofweek("date"))
df = df.withColumn("month", month("date"))
df = df.withColumn("year", year("date"))

df = df.dropna()

print(f"Total rows processed: {df.count()}")
df.show(5)

spark.stop()
