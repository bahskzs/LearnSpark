package org.apache.spark.ml

import org.apache.spark
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, LabeledPoint, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.example.SparkSessionCreate

object RiskWarningDTClassification {


  //val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[*]")
      .appName("MySparkSession").getOrCreate()
    val rfile = spark.read.format("csv").option("header","true").load("src/main/resources/data/tennis_train.tsv")
    val rdd = rfile.rdd
    // rdd.foreach(println)
    val data = rdd.map(line => line.toString().replaceAll("\\[","").
      replaceAll("\\]","").split(",")).map(i=>concat(i))

    //data.saveAsTextFile("src/main/resources/data/sample_libsvm_data4.txt")

    val data2 = spark.read.format("libsvm").load("src/main/resources/data/sample_libsvm_data2.txt")

    val preData = spark.read.format("csv").option("header","true").load("src/main/resources/data/tennis_preview.tsv")

    val preRDD = preData.rdd
    val newData = preRDD.map(line => line.toString().replaceAll("\\[","").
      replaceAll("\\]","").split(",")).map(i=>concat2(i))


    //newData.saveAsTextFile("src/main/resources/data/sample_libsvm_data5.txt")

    newData.foreach(println)


    val data3 = spark.read.format("libsvm").load("src/main/resources/data/sample_libsvm_data5.txt")

    val featureIndexer2 = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(3) // features with > 4 distinct values are treated as continuous.
      .fit(data3)

    println("----------------data3----------------")
    data3.show()


    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data2)

    println("-------------data2----------------")
    data2.show(5)



    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(3) // features with > 4 distinct values are treated as continuous.
      .fit(data2)

    println("-------------data2 fea---------------")
    data2.show(4)

    val Array(trainingData, testData) = data2.randomSplit(Array(0.7, 0.3))

    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setImpurity("entropy")


    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    val predictions_2 = model.transform(data3)

    // predictions_2.write.

    //select("predictedLabel", "label", "features").

    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
    // $example off$



    spark.stop()


  }

  //构造libSVM格式数据,第一列为label
  def concat(a:Array[String]):String ={
    var result=a(0)+" "
    for(i<-1 to a.size.toInt-1) {
      println("--------" + i+ ":--------" +result)
      result=result+i+":"+a(i)
      if(i < a.size.toInt-1){
        result = result + " "
      }

    }
    return result
  }

  def concat2(a:Array[String]):String ={
    var result= "0 "
    for(i<-1 to a.size.toInt-1) {
      result= result+i+":"+a(i)
      if(i < a.size.toInt-1){
        result = result + " "
      }
    }
    return result
  }


}
