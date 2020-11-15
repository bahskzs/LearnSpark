package org.example

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.{SparkConf, SparkContext}

/**
 *
 * @author cat
 * @date 2020-07-18 18:55
 */
object CountArray {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("Hello world")
    val core = new Path("H:\\Desktop\\学习日志\\hadoop\\hadoop\\core-site.xml")
    val hdfs = new Path("H:\\Desktop\\学习日志\\hadoop\\hadoop\\hdfs-site.xml")
    System.setProperty("HADOOP_USER_NAME", "hadoop")
    val sc = new SparkContext(conf)
    sc.hadoopConfiguration.addResource(core)
    sc.hadoopConfiguration.addResource(hdfs)
    val string = Array("Spark is awesome", "Spark is cool", "Hello world")
    val stringRDD = sc.parallelize(string)
    stringRDD.map(l => l).collect.foreach(println)
    stringRDD.saveAsTextFile("hdfs://192.168.41.244:8020/data/wordcount2")

    sc.stop()
  }
}
