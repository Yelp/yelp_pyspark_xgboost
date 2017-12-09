package com.yelp.ads.gbt

import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost}
import ml.dmlc.xgboost4j.scala.spark.{XGBoost, XGBoostModel, XGBoostRegressionModel}
import org.apache.spark.SparkContext
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.apache.spark.ml.feature.{LabeledPoint => MLLabeledPoint}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.commons.logging.{Log, LogFactory}
import org.apache.spark.mllib.regression.LabeledPoint
import scala.collection.JavaConversions._
import java.io._
import java.nio.file.{Files, Paths}
import org.apache.spark.ml.linalg.{DenseVector => MLDenseVector}


object TrainwithRDD {
  private val logger: Log = LogFactory.getLog(this.getClass)
  var rounds = 0

  def train(jsc: JavaSparkContext, trainRDD: JavaRDD[LabeledPoint], paramsInJava: java.util.HashMap[java.lang.String, java.lang.Object]):String={
    /** Train a model with train data (This method will be called from pyspark)
      *
      * @param jsc java spark context
      * @param paramsInJava training parameters for xgboost
      * @return model file name
      */
    logger.info("Training starts")
    val paramsInScala: Map[String, Any] = paramsInJava.toMap
    logger.info("Training parameters are " + paramsInScala)

    val trainRDDScala = trainRDD.rdd.persist(StorageLevel.MEMORY_AND_DISK).
      map{labeledPoint => MLLabeledPoint(
        labeledPoint.label.toDouble,
        labeledPoint.features.asML)}

    _train(jsc.sc, trainRDDScala, paramsInScala.toMap)

  }

  def _train(sparkContext: SparkContext, trainRDDScala: RDD[MLLabeledPoint], params: Map[String, Any]):String={
    /** Train the model with trainRDD
      *
      *  @param sparkContext Spark Context
      *  @param trainRDD Train data
      *  @param params xgboost training parameters more info:
      *                https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
      *  @return model file name on Hadoop file system
      */

    /* RDDs passed to jvm are RDD[org.apache.spark.mllib.regression.LabeledPoint] we need to convert them to
    RDD[org.apache.spark.ml.feature.LabeledPoint] before passing to xgboost
    */

    rounds = params("iterations").asInstanceOf[Int]
    val trainedModel = XGBoost.train(trainRDDScala, params, rounds, nWorkers = 8, useExternalMemory = false)
    val modelFileName = java.io.File.createTempFile("gbt_", "_model").getAbsolutePath()
    logger.info("model file name is " + modelFileName)

    //xgboost training
    var startTime = System.currentTimeMillis
    trainedModel.booster.saveModel(modelFileName)
    logger.info("xgboost training finished in " + (System.currentTimeMillis - startTime) + " ms")

    //xgboost feature importance calculation
    startTime = System.currentTimeMillis
    _saveFeatureImportance(sparkContext, trainedModel, modelFileName)
    logger.info("xgboost feature importance calculation finished in " + (System.currentTimeMillis - startTime) + " ms")

    logger.info("Training is finished.")
    return modelFileName
  }

  def saveModelToS3(jsc: JavaSparkContext, model_path: String, model_s3_path: String){
    /** save xgboost model in path model_path to s3
      *
      * @param jsc java spark context
      * @param model_hdfs_path model path on HDFS
      * @param model_s3_path model path on s3
      */
    logger.info("Transfering model to s3 " + model_s3_path)
    var loadedXGBoostModel = new XGBoostRegressionModel(SXGBoost.loadModel(model_path))
    loadedXGBoostModel.booster.saveModel(model_s3_path)
  }

  def _saveFeatureImportance(sparkContext: SparkContext, model: XGBoostModel, model_path: String){
    /** Save feature importance in a model to file
      *
      * @param sparkContext spark context
      * @param model xgboost model
      * @param model_path model file name
      */
    val scores = model.booster.getFeatureScore();
    try {
      val feature_importance = new File(model_path + "_feature_importance.tsv")
      val bufferWritter = new BufferedWriter(new FileWriter(feature_importance))
      scores.keys.foreach((key) => bufferWritter.write(key + '\t' + scores(key) + '\n'))
      bufferWritter.close()
    }catch {
      case e: FileNotFoundException =>{
        logger.info("Missing file exception")
        e.printStackTrace()
        e.toString()
      }
      case e: IOException => {
        logger.info("IO exception")
        e.printStackTrace()
        e.toString()
      }
    }finally {
      logger.info("Unexpected error in saving feature importance...Exiting finally")
    }
  }

  def predict(jsc: JavaSparkContext, testRDD: JavaRDD[LabeledPoint], model_path: String): String ={
    /** Predict ctr for RDD of labeled points (This mehod will be called from pyspark)
      *
      * @param jsc java spark context
      * @param testRDD java rdd of labeledpoints for test
      * @param model_path path to the model file
      * @return test error
      */
    logger.info("Prediction starts")
    val testRDDScala = testRDD.rdd.
      map{labeledPoint => MLLabeledPoint(
        labeledPoint.label.toDouble,
        labeledPoint.features.asML)}

    var loadedXGBoostModel = new XGBoostRegressionModel(SXGBoost.loadModel(model_path))
    val prediction = loadedXGBoostModel.eval(testRDDScala, evalName = "Test", evalFunc=new EvalLogLoss())
    logger.info("Prediction is ")
    logger.info(prediction)
    return prediction
  }

}