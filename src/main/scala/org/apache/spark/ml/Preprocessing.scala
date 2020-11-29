package org.apache.spark.ml

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.example.SparkSessionCreate

object Preprocessing {

  def main(args: Array[String]): Unit = {
    var trainSample = 1.0
    var testSample = 1.0

    val train = "src/main/resources/data/insurance_train.csv"
    val test = "src/main/resources/data/insurance_test.csv"

    val spark = SparkSessionCreate.createSession()

    import spark.implicits._
    println("Reading data from " + train + " file")

    val trainInput = spark.read.option("header","true").option("inferSchema","true")
      .format("com.databricks.spark.csv").load(train).cache()

    val testInput = spark.read.option("header","true").option("inferSchema","true")
      .format("com.databricks.spark.csv").load(test).cache()

    // 将train.csv拆分为训练集和验证集(75% and 25%)
    println("Preparing data for training model")
    var data = trainInput.withColumnRenamed("loss","label").sample(false, trainSample)

    var DF = data.na.drop()
    if (data == DF){
      println("No null values in the Dataframe")
    }else {
      println("Null values exist in the Dataframe")
      data = DF
    }
    val seed = 12345L
    val splits = data.randomSplit(Array(0.75, 0.25), seed)
    val (trainingData, validationData ) = (splits(0), splits(1))

    trainingData.cache()
    validationData.cache()

    val testData = testInput.sample(false, testSample).cache()

    //识别分类列
    def isCateg(c: String): Boolean = c.startsWith("cat")
    def cateNewCol(c: String): String = if (isCateg(c)) s"idx_${c}" else c

    //删除类别过多的分类
    def removeTooManyCategs(c: String): Boolean = !(c matches "cat(109$|110$|112$|113$|116$)")

    //删除ID列和label列
    def onlyFeatureCols(c: String): Boolean = !(c matches "id|label")

    //构建特征列的确定集
    val featureCols = trainingData.columns
      .filter(removeTooManyCategs)
      .filter(onlyFeatureCols)
      .map(cateNewCol)

    //使用StringIndexer()
    val stringIndexrStages = trainingData.columns.filter(isCateg)
      .map(c => new StringIndexer()
        .setInputCol(c)
        .setOutputCol(cateNewCol(c))
        .fit(trainInput.select(c).union(testInput.select(c))))

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")


    trainingData.write.format("csv").save("src/main/resources/data/insurance_trainingData.csv")

    println(featureCols)
  }




}
