# Databricks notebook source
# DBTITLE 1,Assignment 1 - Prineet Kaur Bhurji 
# Question 1 
# A. Read in the dataset “AB_NYC_2019.json” with an inferred schema.
sc = spark.sparkContext
path = "/FileStore/tables/AB_NYC_2019.JSON"
NYC_AirBnb = spark.read.json(path)
NYC_AirBnb.printSchema()

# COMMAND ----------

# Question 1
# B. Read in the dataset “AB_NYC_2019.csv” using a manually defined schema. Select the appropriate type for the columns.

from pyspark.sql.types import StructField, StructType, StringType, LongType, DoubleType, DateType

myManualSchema = StructType([
  StructField("id", LongType(), False),
  StructField("name", StringType(), True),
  StructField("host_id", LongType(), False),
  StructField("host_name", StringType(), True),
  StructField("neighbourhood_group", StringType(), False),
  StructField("neighbourhood", StringType(), False),
  StructField("latitude", DoubleType(), False),
  StructField("longitude", DoubleType(), False),
  StructField("room_type", StringType(), False),
  StructField("price", LongType(), False),
  StructField("minimum_nights", LongType(), False),
  StructField("number_of_reviews", LongType(), False),
  StructField("last_review", DateType(), True),
  StructField("reviews_per_month", DoubleType(), True),
  StructField("calculated_host_listings_count", LongType(), False),
  StructField("availability_365", LongType(), False)
])

path = "/FileStore/tables/AB_NYC_2019.csv"

NYC_AirBnb = spark\
.read\
.format("csv")\
.option("header","true")\
.schema(myManualSchema)\
.load(path)

NYC_AirBnb.show(3)
NYC_AirBnb.printSchema()

# COMMAND ----------

# Question 2
# A. Get the summary values separately for all numeric columns and all string columns.

NYC_AirBnb.columns

# For numeric values
NYC_AirBnb['id', 'host_id', 'latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'last_review', 'reviews_per_month', 'calculated_host_listings_count',              'availability_365'].describe().show()

# For string values
NYC_AirBnb['name', 'host_name', 'neighbourhood_group', 'neighbourhood', 'room_type'].summary().show()

# COMMAND ----------

# Question 3
# A. Calculate the average price and its standard deviation by room_type and per neighborhood group and neighborhood. Create a statement using SQL, DF, and a grouping sets function.

# NYC_AirBnb.columns
from pyspark.sql.functions import avg, round, col, column, expr, stddev

#Count the number of invoices by customer
# spark.sql("select room_type, avg(price) from NYC_AirBnb group by room_type").show(3)
NYC_AirBnb.groupBy("room_type").agg(avg("price"), stddev("price")).show(3)
# NYC_AirBnb.groupBy("room_type").agg(expr("mean(price)"),expr("stddev(price)")).show(3)

NYC_AirBnb.groupBy("room_type").agg(avg(col("price")),stddev(col("price"))).show()
# NYC_AirBnb.groupBy("room_type").agg(expr("mean(price)"),expr("stddev(price)")).show()
# spark.sql("select room_type, avg(price), std(price) from NYC_AirBnb group by room_type").show(3)

# COMMAND ----------

NYC_AirBnb.groupBy("neighbourhood_group", "neighbourhood").agg(avg("price"), stddev("price")).show(3)
# spark.sql("select neighbourhood_group, neighbourhood, avg(price), std(price) from NYC_AirBnb group by neighbourhood_group, neighbourhood").show(3)

# COMMAND ----------

# Question 4
# B. Handle the missing values in the dataset as follows: for neighbourhood and neighbourhood_group, replace the missing values with ‘NA’; for “availability_365”, replace the missing values with 0.

# NYC_AirBnb.na.fill(0, subset="availability_365").show(2)
# NYC_AirBnb.sort('neighbourhood').fillna("NA", subset="neighbourhood").show(2)

fill_cols_vals = {"neighbourhood": "NA", "neighbourhood_group" : "NA", "availability_365" : 0}
NYC_AirBnb.na.fill(fill_cols_vals).show(2)

# COMMAND ----------

# Question 5
# A. Get the top 3 places where it’s cheapest to stay in Manhattan for 3 days.

NYC_AirBnb.select("id", "name").distinct().where(col("neighbourhood_group") == 'Manhattan').where(col('minimum_nights') >= 2).sort(col('price')).show(3)

NYC_AirBnb.where((col("neighbourhood_group") == "Manhattan") & (col("minimum_nights") >= 2)).select("id", "name").sort(col('price')).show(3)
