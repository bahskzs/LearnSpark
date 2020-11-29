package org.apache.spark.mlib

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
 *
 * @author cat
 * @date 2020-10-05 20:26
 *      url:https://spark.apache.org/docs/latest/mllib-decision-tree.html
 */
object RiskWarningDecisionTreeClassification {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("DecisionTreeClassification")
    val sc = new SparkContext(conf)

    // $example on$
    // Load and parse the data file.
    val data = sc.textFile("hdfs://localhost:9000/data/tennis.tsv")
    val parsedData = data.map {
      line =>  val parts = line.split(',').map(_.toDouble)
        LabeledPoint(parts(parts.length-1), Vectors.dense(parts.init))
    }
    val splits = parsedData.randomSplit(Array(0.7, 0.3),12345L)
    val (trainingData, testData) = (splits(0), splits(1))
    println("---------trainData-----------")
    trainingData.foreach(println)

    println("---------testData-----------")
    testData.foreach(println)

    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 10
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "entropy"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    println("------------result---------------")
    labelAndPreds.foreach(println)

    val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
    println(s"Test Error = $testErr")
    println(s"Learned classification tree model:\n ${model.toDebugString}")

    // Save and load model
    model.save(sc, "hdfs://localhost:9000/tmp/myDTClassificationModel")
    val sameModel = DecisionTreeModel.load(sc, "hdfs://localhost:9000/tmp/myDTClassificationModel")


    val newData = sc.textFile("hdfs://localhost:9000/data/tennis_preview.tsv")
    val parsedNewData = newData.map {
      line =>  val parts = line.split(',').map(_.toDouble)
        (parts(0),Vectors.dense(parts.init))
    }
    parsedNewData.map(l => (l._1,model.predict(l._2)).toString().replaceAll("\\(","").replaceAll("\\)","")
    ).saveAsTextFile("hdfs://localhost:9000/data/result.csv")
    sc.stop()
  }

}