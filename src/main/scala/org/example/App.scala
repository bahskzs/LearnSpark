package org.example

import org.apache.spark
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Hello world!
 *
 */

class App {

}
object App {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("Hello world")
    val sc = new SparkContext(conf)
    val rdd = sc.parallelize(1 to 10)
    val map = rdd.map(_*2)
    map.foreach(x => println(x+" "))
    sc.stop()

  }

}
