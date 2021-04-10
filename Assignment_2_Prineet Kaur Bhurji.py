# Databricks notebook source
# Question 1 - Using the dataset “activity-data”, create a stream that outputs in one table: 
# a) real-time time difference between arrival_time and creation_time aggregated by user and device.
# b) historical minimum and maximum time difference values between arrival_time and creation_time aggregated across all users.
# c) real-time updated difference between real-time time difference and historical minimum and maximum time difference

# COMMAND ----------

filePath = "/FileStore/tables/Streaming"

# COMMAND ----------

staticDF = spark.read\
  .json(filePath)

dataSchema = staticDF.schema

streamingDF = spark.readStream\
  .schema(dataSchema)\
  .option("maxFilesPerTrigger", 1)\
  .json(filePath)

# COMMAND ----------

staticDF.show(4)

# COMMAND ----------

from pyspark.sql.functions import *


histData = staticDF.withColumn("time_diff", (col("Creation_Time")/1000000000).cast("timestamp").cast("double") -                                                                        (col("Arrival_Time")/1000).cast("timestamp").cast("double"))\
                   .select("User","Device","time_diff")\
                   .groupBy("User","Device").agg(min("time_diff").alias("min_diff"), max("time_diff").alias("max_diff"))\
                   .withColumn("hist_diff", col("max_diff")-col("min_diff"))\
                   .select("User", "Device", "hist_diff")


streamDataStream = streamingDF.withColumn("time_diff", (col("Creation_Time")/1000000000).cast("timestamp").cast("double") -                                                                        (col("Arrival_Time")/1000).cast("timestamp").cast("double"))\
  .drop("Arrival_Time", "Creation_Time", "Model", "Index", "gt", "x", "y", "z")\
  .cube("User", "Device").avg()\
  .join(histData, ["User", "Device"])\
  .writeStream\
  .queryName("device_join_counts")\
  .format("memory")\
  .outputMode("complete")\
  .start()

# COMMAND ----------

#Query the table device_join_counts every 5 seconds
from time import sleep
for x in range(5):
    spark.sql("select * from device_join_counts").show(3)
    sleep(5)

# COMMAND ----------

streamDataStream.stop()

# COMMAND ----------

# DBTITLE 0,Untitled
from pyspark.sql.functions import min, max, col
# init your spark dataframe

staticDF.select("gt").distinct().show()

expr = [min(col("avg(time_diff)")),max(col("avg(time_diff)"))]

histData = histData.groupBy("User", "device").agg(*expr).show()

# COMMAND ----------

# Question 2 - Using the dataset “activity-data”, create a stream that outputs in one table the total number of meters user g travels per activity in time intervals of resp. 15 minutes and 30 minutes. Order the table by the most distance travelled per activity. Hint: you can use the columns x, y, z to calculate the distance travelled

# COMMAND ----------

staticDF.show(4)

# COMMAND ----------

from pyspark.sql.functions import sqrt

totalDist = streamingDF.select("User", "x", "y", "z").withColumn("Distance", sqrt(pow((streamingDF['x']), 2) + pow((streamingDF['y']), 2) + pow((streamingDF['y']), 2)))

user_g_dist = totalDist\
  .cube("User").sum("distance")\
  .where("User == 'g'")\
  .writeStream\
  .queryName("user_g_distance")\
  .format("memory")\
  .outputMode("complete")\
  .start()

# COMMAND ----------

from time import sleep
for x in range(5):
    spark.sql("select * from user_g_distance").show(3)
    sleep(5)

# COMMAND ----------

user_g_dist.stop()

# COMMAND ----------

withEventTime = streamingDF\
  .selectExpr("*","cast(cast(Creation_Time as double)/1000000000 as timestamp) as event_time")
