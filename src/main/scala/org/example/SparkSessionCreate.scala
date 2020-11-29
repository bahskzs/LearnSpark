package org.example

import org.apache.spark.sql.SparkSession

object SparkSessionCreate {
  val spark = SparkSessionCreate.createSession()
  def createSession() : SparkSession = {
    val spark = SparkSession.builder().master("local[*]").config("spark.sql.warehouse.dir","H:\\bigdata\\spark-2.4.7-bin-without-hadoop-scala-2.12\\exp\\")
      .appName("MySparkSession").getOrCreate()
    return spark
  }
}
