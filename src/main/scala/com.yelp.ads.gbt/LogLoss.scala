package com.yelp.ads.gbt

import ml.dmlc.xgboost4j.java.XGBoostError
import ml.dmlc.xgboost4j.scala.EvalTrait
import org.apache.commons.logging.LogFactory
import ml.dmlc.xgboost4j.scala.{DMatrix, EvalTrait, ObjectiveTrait, XGBoost}


class EvalLogLoss extends EvalTrait {

  val logger = LogFactory.getLog(classOf[EvalLogLoss])

  private var evalMetric: String = "eval_log_loss"

  /**
    * get evaluate metric
    *
    * @return evalMetric
    */
  override def getMetric: String = evalMetric

  /**
    * evaluate with predicts and data
    *
    * @param predicts predictions as array
    * @param dmat data matrix to evaluate
    * @return result of the metric
    */
  override def eval(predicts: Array[Array[Float]], dmat: DMatrix): Float = {
    var error: Float = 0f
    var labels: Array[Float] = null
    try {
      labels = dmat.getLabel
    } catch {
      case ex: XGBoostError =>
        logger.error(ex)
        return -1f
    }
    val nrow: Int = predicts.length
    for (i <- 0 until nrow) {
      if (labels(i) == 0.0) {
        error += -math.log(1 - predicts(i)(0)).toFloat
      } else if (labels(i) == 1.0) {
        error += -math.log(predicts(i)(0)).toFloat
      }
    }
    error / labels.length
  }
}
