/*
 * Copyright [2013-2015] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.dtrain;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.guagua.util.FileUtils;
import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.util.CommonUtils;

import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class TreeModelEvalAndScoreTest2 {

    private IndependentTreeModel iTreeModel;

    @BeforeClass
    public void setUp() throws IOException {
        String modelPath = "src/test/resources/dttest/model/tm.gbt";
        FileInputStream fi = null;
        try {
            fi = new FileInputStream(modelPath);
            iTreeModel = IndependentTreeModel.loadFromStream(fi, true);
        } finally {
            fi.close();
        }
    }

    @Test
    public void testEvalScore() throws IOException {
        List<String> lines = FileUtils.readLines(new File("src/test/resources/dttest/data/tm.csv"));
        if(lines.size() <= 1) {
            
            return;
        }
        String[] headers = CommonUtils.split(lines.get(0), "|");
        // score with format <String, String>
        for(int i = 1; i < lines.size(); i++) {
            Map<String, String> map = new HashMap<String, String>();
            Map<String, Object> mapObj = new HashMap<String, Object>();

            String[] data = CommonUtils.split(lines.get(i), "|");;
            // System.out.println("data len is " + data.length);
            if(data.length != headers.length) {
                System.out.println("One invalid input data");
                break;
            }
            for(int j = 0; j < headers.length; j++) {
                map.put(headers[j], data[j]);
                mapObj.put(headers[j], data[j]);
            }
            double[] scores = iTreeModel.compute(mapObj);

            System.out.println(" shifu score: " + scores[0]);
        }

    }

}
