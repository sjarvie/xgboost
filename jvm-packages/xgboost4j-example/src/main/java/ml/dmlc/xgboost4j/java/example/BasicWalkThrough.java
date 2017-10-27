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
package ml.dmlc.xgboost4j.java.example;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import ml.dmlc.xgboost4j.scala.Classification;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;

/**
 * a simple example of java wrapper for xgboost
 *
 * @author hzx
 */
public class BasicWalkThrough {
  public static boolean checkPredicts(float[][] fPredicts, float[][] sPredicts) {
    if (fPredicts.length != sPredicts.length) {
      return false;
    }

    for (int i = 0; i < fPredicts.length; i++) {
      if (!Arrays.equals(fPredicts[i], sPredicts[i])) {
        return false;
      }
    }

    return true;
  }

  public static void saveDumpModel(String modelPath, String[] modelInfos) throws IOException {
    try{
      PrintWriter writer = new PrintWriter(modelPath, "UTF-8");
      for(int i = 0; i < modelInfos.length; ++ i) {
        writer.print("booster[" + i + "]:\n");
        writer.print(modelInfos[i]);
      }
      writer.close();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) throws IOException, XGBoostError {
    // load file from text file, also binary buffer generated by xgboost4j
    DMatrix trainMat = new DMatrix(Classification.trainFile().toString());
    DMatrix testMat = new DMatrix(Classification.testFile().toString());

    HashMap<String, Object> params = new HashMap<String, Object>();
    params.put("eta", 1.0);
    params.put("max_depth", 2);
    params.put("silent", 1);
    params.put("objective", "binary:logistic");


    HashMap<String, DMatrix> watches = new HashMap<String, DMatrix>();
    watches.put("train", trainMat);
    watches.put("test", testMat);

    //set round
    int round = 2;

    //train a boost model
    Booster booster = XGBoost.train(trainMat, params, round, watches, null, null);

    //predict
    float[][] predicts = booster.predict(testMat);

    //save model to modelPath
    File file = new File("./model");
    if (!file.exists()) {
      file.mkdirs();
    }

    String modelPath = "./model/xgb.model";
    booster.saveModel(modelPath);

    //dump model with feature map
    String[] modelInfos = booster.getModelDump("../../demo/data/featmap.txt", false);
    saveDumpModel("./model/dump.raw.txt", modelInfos);

    //save dmatrix into binary buffer
    testMat.saveBinary("./model/dtest.buffer");

    //reload model and data
    Booster booster2 = XGBoost.loadModel("./model/xgb.model");
    DMatrix testMat2 = new DMatrix("./model/dtest.buffer");
    float[][] predicts2 = booster2.predict(testMat2);

    //check the two predicts
    System.out.println(checkPredicts(predicts, predicts2));
  }
}
