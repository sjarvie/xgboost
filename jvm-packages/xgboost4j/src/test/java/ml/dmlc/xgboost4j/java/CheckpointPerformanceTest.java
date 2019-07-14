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
package ml.dmlc.xgboost4j.java;

import java.io.*;
import java.util.Map;
import java.util.HashMap;

import org.junit.Test;


public class CheckpointPerformanceTest {
    @Test
    public void testTrainFromExistingModel() throws XGBoostError, IOException {
      DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
      DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");
  
      Map<String, Object> paramMap = new HashMap<String, Object>() {
        {
          put("eta", 1.0);
          put("max_depth", 2);
          put("verbosity", 3);
          put("objective", "binary:logistic");
        }
      };
  
      //set empty watchList
      HashMap<String, DMatrix> watches = new HashMap<String, DMatrix>();
  
  //    watches.put("train", trainMat);
  //    watches.put("test", testMat);
  
      // Train without saving temp booster
      int round = 401;
      long start = System.currentTimeMillis();
      Booster booster1 = XGBoost.train(trainMat, paramMap, round, watches, null, null, null, 0);
      long timeForAll = System.currentTimeMillis() - start;
      System.err.printf("Time to train everything: %d\n", timeForAll);
  
      // Train with temp Booster
      round = 200;
      start = System.currentTimeMillis();
      Booster tempBooster = XGBoost.train(trainMat, paramMap, round, watches, null, null, null, 0);
      long timeForFirstHalf = System.currentTimeMillis() - start;
      System.err.printf("Time to train the first half: %d\n", timeForFirstHalf);
  
      DMatrix[] allMats = new DMatrix[2];
      allMats[0] = trainMat;
      allMats[1] = testMat;
  
      // Save tempBooster to bytestream and load back
      int prevVersion = tempBooster.getVersion();
      ByteArrayInputStream in = new ByteArrayInputStream(tempBooster.toByteArray());
      //byte[] byteArray = tempBooster.toByteArray();
      //tempBooster = new Booster(paramMap, allMats);
      //tempBooster.loadFromByteArray(byteArray);
      tempBooster.dispose();
      tempBooster = XGBoost.loadModel(in);
      tempBooster.setVersion(prevVersion);
  
      // Continue training using tempBooster
      start = System.currentTimeMillis();
      Booster tempBoosterPlusOne = XGBoost.train(trainMat, paramMap, round + 1, watches, null, null, null, 0, tempBooster);
      long timeForFirstRoundAfterSave = System.currentTimeMillis() - start;
      System.err.printf("Time to train one iteration after checkpoint: %d\n", timeForFirstRoundAfterSave);
  
      start = System.currentTimeMillis();
      Booster booster2 = XGBoost.train(trainMat, paramMap, 2 * round + 1, watches, null, null, null, 0, tempBoosterPlusOne);
      long timeForSecondHalf = System.currentTimeMillis() - start;
      System.err.printf("Time to train the second half: %d\n", timeForSecondHalf);

      booster2.dispose();
    }  
}