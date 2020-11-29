package org.example

import org.apache.hadoop.fs.Path
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.sql.SparkSession
/**
 *
 * @author cat
 * @date 2020-09-14 15:30
 */
object DecisionTreeDemo {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("Decision Tree")
    val sc = new SparkContext(conf)
    println("load files ...")
    val data = sc.textFile("src/main/resources/data/tennis.csv")
    val parsedData = data.map {
      line =>  val parts = line.split(',').map(_.toDouble)
      LabeledPoint(parts(0), Vectors.dense(parts.tail))
    }
    val model = DecisionTree.train(parsedData, Classification, Entropy, 3)
    val v=Vectors.dense(0.0,1.0,0.0)

    println("v --- "+v);

    val v1=Vectors.dense(0.0,1.0,1.0)
    println("predict1:1.0,1.0,0.0----" + model.predict(v))
    println("predict2:0.0,0.0,0.0----" + model.predict(v1))
    println(model.toDebugString)

    sc.stop()
  }

}
