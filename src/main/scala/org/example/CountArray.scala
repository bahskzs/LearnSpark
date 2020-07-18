package org.example

import org.apache.spark.{SparkConf, SparkContext}

/**
 *
 * @author cat
 * @date 2020-07-18 18:55
 */
object CountArray {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("Hello world")
    val sc = new SparkContext(conf)
    val string = Array("Spark is awesome", "Spark is cool")
    val stringRDD = sc.parallelize(string)
    stringRDD.map(l => l).collect.foreach(println)
    sc.stop()
  }
}
