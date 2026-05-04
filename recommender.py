"""
recommender.py
==============
Big Data Analytics — Movie Recommendation System
================================================
Algorithm  : Collaborative Filtering using Apache Spark MLlib ALS
             (Alternating Least Squares)
Dataset    : MovieLens 100K  (943 users | 1,682 movies | 100,000 ratings)
Source     : GroupLens Research, University of Minnesota
             https://grouplens.org/datasets/movielens/100k/

BDA Concepts Demonstrated:
  - Distributed in-memory computing (Apache Spark)
  - Lazy evaluation and DAG execution
  - Spark DataFrames & SQL operations
  - MLlib pipeline (ALS Collaborative Filtering)
  - RDD transformations and actions
  - Partitioned data shuffling
  - RMSE / MAE model evaluation

Run : python recommender.py
"""

import os, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────
DATA_DIR   = os.path.join("data", "ml-100k")
OUTPUT_DIR = "output"
ALS_PARAMS = dict(rank=20, maxIter=15, regParam=0.1,
                  coldStartStrategy="drop", implicitPrefs=False, seed=42)
TOP_N      = 10
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────
#  0.  CHECK DATASET
# ─────────────────────────────────────────────────────
def _check_data():
    udata = os.path.join(DATA_DIR, "u.data")
    if not os.path.exists(udata):
        print("[!] Dataset not found.  Run: python download_data.py")
        raise SystemExit(1)

_check_data()


# ─────────────────────────────────────────────────────
#  1.  SPARK SESSION
# ─────────────────────────────────────────────────────
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import (col, explode, count, avg, desc,
                                   stddev, min as spark_min, max as spark_max,
                                   year, udf, when)
from pyspark.sql.types import (StructType, StructField,
                               IntegerType, FloatType, StringType, LongType)

print("\n" + "=" * 65)
print("   BDA Movie Recommendation System — Apache Spark MLlib ALS")
print("   Dataset : MovieLens 100K  |  943 users  |  1,682 movies")
print("=" * 65)

spark = (SparkSession.builder
         .appName("BDA_MovieRecommendation_ALS")
         .master("local[*]")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.driver.memory", "2g")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
print("\n[1] Spark Session initialised  ✓")
print(f"    Spark version : {spark.version}")
print(f"    Partitions    : 8  (local[*])")


# ─────────────────────────────────────────────────────
#  2.  LOAD MOVIELENS 100K
# ─────────────────────────────────────────────────────
print("\n[2] Loading MovieLens 100K Dataset ...")

# ── ratings  (userId | movieId | rating | timestamp)
ratings_schema = StructType([
    StructField("userId",    IntegerType(), False),
    StructField("movieId",   IntegerType(), False),
    StructField("rating",    FloatType(),   False),
    StructField("timestamp", LongType(),    False),
])
ratings_df = (spark.read.csv(
    os.path.join(DATA_DIR, "u.data"),
    schema=ratings_schema, sep="\t")
    .drop("timestamp"))

# ── movies  (movieId | title | release_date | video_release | url | genre_flags...)
raw_movies = pd.read_csv(
    os.path.join(DATA_DIR, "u.item"),
    sep="|", encoding="latin-1", header=None,
    names=["movieId","title","release_date","video_release","url",
           "unknown","Action","Adventure","Animation","Childrens","Comedy",
           "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",
           "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"])

GENRES = ["Action","Adventure","Animation","Childrens","Comedy","Crime",
          "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
          "Mystery","Romance","Sci-Fi","Thriller","War","Western"]

def get_genres(row):
    gs = [g for g in GENRES if row.get(g, 0) == 1]
    return "|".join(gs) if gs else "Unknown"

raw_movies["genres"] = raw_movies.apply(get_genres, axis=1)
raw_movies["primary_genre"] = raw_movies["genres"].apply(lambda x: x.split("|")[0])
raw_movies["year"] = (raw_movies["release_date"]
                      .str.extract(r"(\d{4})")[0]
                      .fillna("Unknown"))

movies_pd = raw_movies[["movieId", "title", "genres", "primary_genre", "year"]].copy()
movies_pd["movieId"] = movies_pd["movieId"].astype(int)
movies_df = spark.createDataFrame(movies_pd)

# ── users  (userId | age | gender | occupation | zip)
users_schema = StructType([
    StructField("userId",     IntegerType(), False),
    StructField("age",        IntegerType(), False),
    StructField("gender",     StringType(),  False),
    StructField("occupation", StringType(),  False),
    StructField("zip",        StringType(),  False),
])
users_df = (spark.read.csv(
    os.path.join(DATA_DIR, "u.user"),
    schema=users_schema, sep="|"))

rc = ratings_df.count()
mc = movies_df.count()
uc = users_df.count()
density = rc / (uc * mc) * 100
print(f"    ✓ {rc:,} ratings | {uc:,} users | {mc:,} movies")
print(f"    ✓ Matrix density : {density:.2f}%  (typical: 1–10%)")


# ─────────────────────────────────────────────────────
#  3.  EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────
print("\n[3] Exploratory Data Analysis")
print("-" * 45)

# Rating distribution
rating_dist = (ratings_df.groupBy("rating")
               .count().orderBy("rating").toPandas())
print("\n    Rating Distribution:")
for _, r in rating_dist.iterrows():
    bar = "█" * int(r["count"] // 2000)
    print(f"    ★{r['rating']:.1f}  {bar:<25}  {int(r['count']):,}")

# Top 10 most-rated movies
top_movies = (ratings_df
    .groupBy("movieId")
    .agg(count("rating").alias("num_ratings"),
         avg("rating").alias("avg_rating"),
         stddev("rating").alias("std_rating"))
    .join(movies_df, "movieId")
    .orderBy(desc("num_ratings"))
    .select("title","primary_genre","num_ratings","avg_rating","std_rating")
    .limit(10).toPandas())
print("\n    Top 10 Most-Rated Movies:")
for _, r in top_movies.iterrows():
    print(f"    {r['title']:<40}  {int(r['num_ratings'])} ratings  "
          f"★{r['avg_rating']:.2f}")

# Genre stats
genre_stats = (ratings_df
    .join(movies_df, "movieId")
    .groupBy("primary_genre")
    .agg(count("rating").alias("total_ratings"),
         avg("rating").alias("avg_rating"),
         count("movieId").alias("num_movies"))
    .orderBy(desc("total_ratings")).toPandas())
print("\n    Ratings per Genre:")
for _, r in genre_stats.iterrows():
    bar = "█" * int(r["total_ratings"] // 1500)
    print(f"    {r['primary_genre']:<14}  {bar:<15}  "
          f"{int(r['total_ratings']):,} ratings  ★{r['avg_rating']:.2f}")

# User stats
user_stats = (ratings_df
    .groupBy("userId")
    .agg(count("rating").alias("num_ratings"),
         avg("rating").alias("avg_rating"))
    .toPandas())
print(f"\n    User Activity (ratings per user):")
print(f"    Min: {user_stats['num_ratings'].min():.0f}  "
      f"Max: {user_stats['num_ratings'].max():.0f}  "
      f"Mean: {user_stats['num_ratings'].mean():.1f}  "
      f"Median: {user_stats['num_ratings'].median():.1f}")


# ─────────────────────────────────────────────────────
#  4.  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────
train_df, test_df = ratings_df.randomSplit([0.8, 0.2], seed=RANDOM_SEED)
tc, vc = train_df.count(), test_df.count()
print(f"\n[4] Train/Test Split")
print(f"    Train : {tc:,} ratings  ({tc/(tc+vc)*100:.0f}%)")
print(f"    Test  : {vc:,} ratings  ({vc/(tc+vc)*100:.0f}%)")


# ─────────────────────────────────────────────────────
#  5.  TRAIN ALS MODEL
# ─────────────────────────────────────────────────────
print(f"\n[5] Training ALS Model ...")
print(f"    rank={ALS_PARAMS['rank']}  |  maxIter={ALS_PARAMS['maxIter']}  "
      f"|  regParam={ALS_PARAMS['regParam']}")

t0 = time.time()
als = ALS(
    rank=ALS_PARAMS["rank"],
    maxIter=ALS_PARAMS["maxIter"],
    regParam=ALS_PARAMS["regParam"],
    userCol="userId", itemCol="movieId", ratingCol="rating",
    coldStartStrategy=ALS_PARAMS["coldStartStrategy"],
    implicitPrefs=ALS_PARAMS["implicitPrefs"],
    seed=ALS_PARAMS["seed"],
)
model = als.fit(train_df)
elapsed = time.time() - t0
print(f"    ✓ Training complete in {elapsed:.1f}s")


# ─────────────────────────────────────────────────────
#  6.  MODEL EVALUATION
# ─────────────────────────────────────────────────────
print(f"\n[6] Model Evaluation")
print("-" * 45)

predictions = model.transform(test_df).dropna(subset=["prediction"])

def evaluate(metric):
    return RegressionEvaluator(
        metricName=metric, labelCol="rating",
        predictionCol="prediction").evaluate(predictions)

rmse = evaluate("rmse")
mae  = evaluate("mae")
r2   = evaluate("r2")

print(f"    RMSE : {rmse:.4f}  (Root Mean Squared Error)")
print(f"    MAE  : {mae:.4f}  (Mean Absolute Error)")
print(f"    R²   : {r2:.4f}  (Coefficient of Determination)")
print(f"\n    On average, predictions are off by ±{mae:.2f} stars (1–5 scale).")


# ─────────────────────────────────────────────────────
#  7.  GENERATE TOP-N RECOMMENDATIONS
# ─────────────────────────────────────────────────────
print(f"\n[7] Generating Top-{TOP_N} Recommendations per User ...")
t1 = time.time()
user_recs = model.recommendForAllUsers(TOP_N)
elapsed2  = time.time() - t1
print(f"    ✓ Done in {elapsed2:.1f}s for all {uc:,} users")

recs_flat = (user_recs
    .select(col("userId"), explode(col("recommendations")).alias("rec"))
    .select(col("userId"),
            col("rec.movieId").alias("movieId"),
            col("rec.rating").alias("predicted_rating"))
    .join(movies_df, "movieId")
    .select("userId","movieId","title","primary_genre","genres","predicted_rating")
    .orderBy("userId", col("predicted_rating").desc()))

# Sample for 5 users
sample_users = [1, 42, 100, 250, 500]
sample_recs = (recs_flat
    .filter(col("userId").isin(sample_users))
    .toPandas())

print(f"\n    Sample Recommendations (users {sample_users}):")
print("    " + "-" * 55)
cur = None
for _, r in sample_recs.iterrows():
    if r["userId"] != cur:
        cur = r["userId"]; print(f"\n    User {cur}:")
    sc = min(5.0, round(float(r["predicted_rating"]), 1))
    print(f"      {'★'*round(sc):<5}  {r['title']:<38}  ({r['primary_genre']})  {sc}")


# ─────────────────────────────────────────────────────
#  8.  ITEM-ITEM SIMILARITY (ALS Item Factors)
# ─────────────────────────────────────────────────────
print(f"\n[8] Item-Item Collaborative Filtering ...")
item_recs = model.recommendForAllItems(TOP_N)
item_recs_flat = (item_recs
    .select(col("movieId").alias("source_movie_id"),
            explode(col("recommendations")).alias("rec"))
    .select(col("source_movie_id"),
            col("rec.userId").alias("similar_movie_id"),
            col("rec.rating").alias("similarity_score"))
    .join(movies_df.withColumnRenamed("movieId","source_movie_id")
                   .withColumnRenamed("title","source_title"), "source_movie_id")
    .select("source_movie_id","source_title","similar_movie_id","similarity_score")
    .orderBy("source_movie_id", col("similarity_score").desc()))
print("    ✓ Item-item similarity matrix computed")


# ─────────────────────────────────────────────────────
#  9.  SAVE ALL OUTPUTS
# ─────────────────────────────────────────────────────
print(f"\n[9] Saving Outputs to '{OUTPUT_DIR}/' ...")

# Save full recommendation list (all users)
all_recs_pd = recs_flat.toPandas()
all_recs_pd.to_csv(f"{OUTPUT_DIR}/all_recommendations.csv", index=False)

# Save sample recommendations
sample_recs.to_csv(f"{OUTPUT_DIR}/sample_recommendations.csv", index=False)

# Save top movies stats
top_movies.to_csv(f"{OUTPUT_DIR}/top_movies.csv", index=False)

# Save genre stats
genre_stats.to_csv(f"{OUTPUT_DIR}/genre_stats.csv", index=False)

# Save rating distribution
rating_dist.to_csv(f"{OUTPUT_DIR}/rating_distribution.csv", index=False)

# Save user stats
user_stats.to_csv(f"{OUTPUT_DIR}/user_stats.csv", index=False)

# Save all movies
movies_pd.to_csv(f"{OUTPUT_DIR}/movies.csv", index=False)

# Save users
users_df.toPandas().to_csv(f"{OUTPUT_DIR}/users.csv", index=False)

# Save model metrics
pd.DataFrame([{
    "RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4),
    "num_users": int(uc), "num_movies": int(mc), "num_ratings": int(rc),
    "train_ratings": int(tc), "test_ratings": int(vc),
    "rank": ALS_PARAMS["rank"], "maxIter": ALS_PARAMS["maxIter"],
    "regParam": ALS_PARAMS["regParam"], "training_time_sec": round(elapsed, 1),
}]).to_csv(f"{OUTPUT_DIR}/model_metrics.csv", index=False)

outputs = ["all_recommendations.csv","sample_recommendations.csv","top_movies.csv",
           "genre_stats.csv","rating_distribution.csv","user_stats.csv",
           "movies.csv","users.csv","model_metrics.csv"]
for f in outputs:
    path = f"{OUTPUT_DIR}/{f}"
    size = os.path.getsize(path)
    print(f"    ✓ {f:<40} {size:>10,} bytes")

spark.stop()
print("\n" + "=" * 65)
print("   SUMMARY")
print("=" * 65)
print(f"   Dataset     : MovieLens 100K — GroupLens Research")
print(f"   Records     : {rc:,} ratings  |  {uc} users  |  {mc} movies")
print(f"   Algorithm   : ALS Collaborative Filtering (Spark MLlib)")
print(f"   RMSE        : {rmse:.4f}")
print(f"   MAE         : {mae:.4f}")
print(f"   R²          : {r2:.4f}")
print(f"   Output      : Top-{TOP_N} recommendations for all {uc} users saved")
print("=" * 65)
print("\n✓ All done!  Next: streamlit run app.py\n")
