import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, Imputer, StandardScaler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}

object MyApp {

  // driver code
  def main(args: Array[String]): Unit = {
    val myApp = new MyApp()
    myApp.run()
  }
}

class MyApp() {

  val dataset_path = "src/main/scala/datasets/framingham_heart_disease.csv";

  // intialising spark builder
  val spark = SparkSession.builder()
    .master("local[*]")
    .appName("DataScience")
    .getOrCreate()

  def run(): Unit = {

    // load datasets
    var df = spark.read.format("csv").option("header", "true").load(dataset_path)
    df = df.drop("education")
    df.show()

    val indexedDF = prepareData(df)
    indexedDF = indexedDF.na.fill(0)

    val im = new Imputer().setInputCols(indexedDF.columns).setOutputCols(indexedDF.columns).setStrategy("mean")

    val imputedDataFrame = im.fit(indexedDF).transform(indexedDF)

    // Use VectorAssembler to combine all the features into a single vector column
    val assembler = new VectorAssembler()
      .setInputCols(Array("male_index", "currentSmoker_index", "cigsPerDay_index", "BPMeds_index", "prevalentStroke_index",
       "prevalentHyp_index", "diabetes_index", "totChol_index", "sysBP_index", "diaBP_index", "BMI_index", "heartRate_index", "glucose_index"))
      .setOutputCol("features")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithMean(true)
      .setWithStd(true)

    val pipeline = new Pipeline().setStages(Array(assembler, scaler))
    val scaledDF = pipeline.fit(imputedDataFrame).transform(imputedDataFrame)
    scaledDF.show()

    val Array(trainDF, testDF) = scaledDF.randomSplit(Array(0.6, 0.4), seed = 50)

    val lr = new LogisticRegression()
      .setFeaturesCol("scaledFeatures")
      .setLabelCol("TenYearCHD_index")
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    val lrModel = lr.fit(trainDF)
    val predictions = lrModel.transform(testDF)

    val binaryEvaluator = new BinaryClassificationEvaluator().setRawPredictionCol("rawPrediction").setLabelCol("TenYearCHD_index")
    println(s"Evaluation: ${binaryEvaluator.evaluate(predictions)}")
  }

   // This function filters and prepares data for training Model
  def prepareData(df: DataFrame): DataFrame = {
    val indexers = Array(
      "male", "currentSmoker", "cigsPerDay", "BPMeds", "prevalentStroke", "prevalentHyp",
      "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose", "TenYearCHD"
    ).map(col => new StringIndexer().setInputCol(col).setOutputCol(col+"_index"))
    val indexerPipeline = new Pipeline().setStages(indexers)
    indexerPipeline.fit(df).transform(df)
      .drop("male", "age", "currentSmoker", "cigsPerDay", "BPMeds", "prevalentStroke",
        "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate",
        "glucose", "TenYearCHD")
  }
}

