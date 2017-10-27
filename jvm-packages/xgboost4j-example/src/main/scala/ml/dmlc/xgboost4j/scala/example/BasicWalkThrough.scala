/*
 Copyright (c) 2014 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package ml.dmlc.xgboost4j.scala.example

import java.io.{File, PrintWriter}

import scala.collection.mutable

import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix}
import ml.dmlc.xgboost4j.scala.{Classification, DMatrix, XGBoost}

object BasicWalkThrough {
  def saveDumpModel(modelPath: String, modelInfos: Array[String]): Unit = {
    val writer = new PrintWriter(modelPath, "UTF-8")
    for (i <- 0 until modelInfos.length) {
      writer.print(s"booster[$i]:\n")
      writer.print(modelInfos(i))
    }
    writer.close()
  }

  def main(args: Array[String]): Unit = {
    val trainMax = new DMatrix(Classification.trainFile.toString)
    val testMax = new DMatrix(Classification.testFile.toString)

    val params = new mutable.HashMap[String, Any]()
    params += "eta" -> 1.0
    params += "max_depth" -> 2
    params += "silent" -> 1
    params += "objective" -> "binary:logistic"

    val watches = new mutable.HashMap[String, DMatrix]
    watches += "train" -> trainMax
    watches += "test" -> testMax

    val round = 2
    // train a model
    val booster = XGBoost.train(trainMax, params.toMap, round, watches.toMap)
    // predict
    val predicts = booster.predict(testMax)
    // save model to model path
    val file = new File("./model")
    if (!file.exists()) {
      file.mkdirs()
    }
    booster.saveModel(file.getAbsolutePath + "/xgb.model")
    // dump model with feature map
    val modelInfos = booster.getModelDump(file.getAbsolutePath + "/featmap.txt", false)
    saveDumpModel(file.getAbsolutePath + "/dump.raw.txt", modelInfos)
    // save dmatrix into binary buffer
    testMax.saveBinary(file.getAbsolutePath + "/dtest.buffer")

    // reload model and data
    val booster2 = XGBoost.loadModel(file.getAbsolutePath + "/xgb.model")
    val testMax2 = new DMatrix(file.getAbsolutePath + "/dtest.buffer")
    val predicts2 = booster2.predict(testMax2)

    // check predicts
    println(checkPredicts(predicts, predicts2))
  }

  def checkPredicts(fPredicts: Array[Array[Float]], sPredicts: Array[Array[Float]]): Boolean = {
    require(fPredicts.length == sPredicts.length, "the comparing predicts must be with the same " +
      "length")
    for (i <- fPredicts.indices) {
      if (!java.util.Arrays.equals(fPredicts(i), sPredicts(i))) {
        return false
      }
    }
    true
  }
}
