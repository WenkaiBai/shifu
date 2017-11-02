/*
 * Copyright [2013-2017] PayPal Software Foundation
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
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.NoSuchFileException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import ml.shifu.guagua.util.FileUtils;
import ml.shifu.shifu.core.dtrain.nn.TFNNModel;
import ml.shifu.shifu.core.processor.ExportModelProcessor;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.testng.annotations.Test;

/**
 * @author pengzhang
 * 
 */
public class TFNNSpec2Test {

    @Test
    public void exportTf2ShifuModel() throws IOException, URISyntaxException {
        ClassLoader classLoader = (this.getClass()).getClassLoader();

        File hadoopWinUtilFile = getAbsoluteFolder(classLoader, "hadoop-2.6.5");
        System.setProperty("hadoop.home.dir", hadoopWinUtilFile.toString());

        File modelFile = getModelFile(classLoader, "tfmodel/spec2/saved_model.pb");

        TFNNModel tfNNModel = ExportModelProcessor.createFromTfToEncog(modelFile.getParent().toString(),
                "dense_2/Sigmoid", "serve");
        Configuration conf = new Configuration();

        Path output = new Path(".", "modelspec2.tfnn");
        TFNNModel.save(tfNNModel.getColumnIndexNameMap(), tfNNModel.getColumnTypeMap(),
                tfNNModel.getOneHotCategoryMap(), tfNNModel.getBasicNetwork(), FileSystem.getLocal(conf), output);
    }

    private File getModelFile(ClassLoader classLoader, String modelFile) throws NoSuchFileException, URISyntaxException {
        String protoPath = modelFile;
        URL protoResource = classLoader.getResource(protoPath);
        if(protoResource == null) {
            throw new NoSuchFileException(protoPath);
        }

        File protoFile = (Paths.get(protoResource.toURI())).toFile();
        return protoFile;
    }

    private File getAbsoluteFolder(ClassLoader classLoader, String currFolder) throws NoSuchFileException,
            URISyntaxException {
        String protoPath = currFolder;
        URL protoResource = classLoader.getResource(protoPath);
        if(protoResource == null) {
            throw new NoSuchFileException(protoPath);
        }
        File protoFile = (Paths.get(protoResource.toURI())).toFile();
        return protoFile;
    }

    @Test
    public void testModelScore() throws IOException {
        String modelPath = "src/test/resources/tfmodel/modelspec2.tfnn";
        FileInputStream fi = null;
        TFNNModel tfNNModel;
        try {
            fi = new FileInputStream(modelPath);
            // long start = System.nanoTime();
            tfNNModel = TFNNModel.loadFromStream(fi);
        } finally {
            fi.close();
        }
        List<String> lines = FileUtils.readLines(new File("src/test/resources/tfmodel/spec2.csv"));

        if(lines.size() <= 1) {
            return;
        }
        String[] headers = CommonUtils.split(lines.get(0), "|");
        // score with format <String, String>
        for(int i = 1; i < lines.size(); i++) {
            String[] data = CommonUtils.split(lines.get(i), "|");;
            if(data.length != headers.length) {
                System.out.println("One invalid input data");
                break;
            }

            int size = tfNNModel.getBasicNetwork().getInputCount();
            double[] inputs = new double[size];
            for(int j = 2; j < data.length; j++) {
                inputs[j - 2] = Double.parseDouble(data[j]);
            }

            double[] scores = tfNNModel.compute(inputs);
            System.out.println("Score is " + Arrays.toString(scores));
        }

    }
}
