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
package ml.dmlc.xgboost4j.scala.spark

import java.nio.file.Files

import org.apache.spark.TaskContext
import org.scalatest.FunSuite

import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala._

class XGBoostCheckpointPerformanceSuite extends FunSuite with PerTest {
  test("performance problem with checkpoints") {
    val eval = new EvalError()
    val training = buildDataFrame(Classification.train)

    val checkpointInterval = 1000

    val tmpPath = Files.createTempDirectory("checkpoint_perf").toAbsolutePath.toString
    val paramMap = Map("eta" -> "1", "max_depth" -> 2,
        "objective" -> "binary:logistic", "checkpoint_path" -> tmpPath,
        "checkpoint_interval" -> checkpointInterval, "num_workers" -> numWorkers,
        "verbosity" -> 1, "tree_method" -> "approx")

    val startFirstRound = System.currentTimeMillis()
    val prevModel = new XGBoostClassifier(paramMap
        ++ Seq("num_round" -> s"${checkpointInterval + 1}")).fit(training)
    val finishFirstRound = System.currentTimeMillis()
    info((finishFirstRound-startFirstRound).toString)

    for (i <- 0 until 1) {
        val startSecondRound = System.currentTimeMillis()
        // Train next model based on prev model
        val nextModel = new XGBoostClassifier(paramMap
        ++ Seq("num_round" -> s"${2 * checkpointInterval - 1}")).fit(training)
        val finishSecondRound = System.currentTimeMillis()
        info((finishSecondRound - startSecondRound).toString)
    }
  }
}
