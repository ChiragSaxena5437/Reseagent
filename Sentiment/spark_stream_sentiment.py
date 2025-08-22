from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import StructType, StringType
from textblob import TextBlob
import os
import sys

# ---------------------------
# 1. Spark session
# ---------------------------
# Point Spark to your venv’s python
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = SparkSession.builder \
    .appName("SentimentStream") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ---------------------------
# 2. Schema for incoming JSON
# ---------------------------
schema = StructType() \
    .add("username", StringType()) \
    .add("content", StringType())

# ---------------------------
# 3. Read socket stream
# ---------------------------
raw_stream = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# ---------------------------
# 4. Parse JSON
# ---------------------------
parsed = raw_stream.select(from_json(col("value"), schema).alias("data")).select("data.*")

# ---------------------------
# 5. Sentiment function
# ---------------------------
def get_sentiment(text):
    if text is None or text.strip() == "":
        return "Neutral"
    analysis = TextBlob(text).sentiment.polarity
    if analysis > 0:
        return "Positive"
    elif analysis < 0:
        return "Negative"
    else:
        return "Neutral"

sentiment_udf = udf(get_sentiment, StringType())

# Add sentiment column
with_sentiment = parsed.withColumn("sentiment", sentiment_udf(col("content")))

# ---------------------------
# 6. Write stream to console
# ---------------------------
query = with_sentiment.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()
