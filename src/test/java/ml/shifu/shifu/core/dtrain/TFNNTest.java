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
import java.net.URLClassLoader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.tensorflow.TensorFlow;
import org.testng.annotations.Test;

import ml.shifu.guagua.util.FileUtils;
import ml.shifu.shifu.core.dtrain.nn.TFNNModel;
import ml.shifu.shifu.core.processor.ExportModelProcessor;
import ml.shifu.shifu.util.CommonUtils;

/**
 * @author pengzhang
 * 
 */
public class TFNNTest {

	@Test
	public void validate() throws IOException {
		System.out.println(TensorFlow.class.getClassLoader().getResourceAsStream("ml/shifu/shifu/ShifuCLI.class"));
		System.out.println(TensorFlow.class.getClassLoader()
				.getResourceAsStream("org/tensorflow/native/windows-x86/tensorflow_jni.dll"));

		System.out.println(Arrays.toString(((URLClassLoader) TensorFlow.class.getClassLoader()).getURLs()));
		System.setProperty("hadoop.home.dir", "D:\\Programs\\hadoop-2.6.5");

		TFNNModel tfNNModel = ExportModelProcessor.createFromTfToEncog(
				new Path("D:\\workspace\\eclipse-shifu-new\\shifu\\src\\test\\resources\\tfmodel", "DNNRegressionAuto").toString(), ExportModelProcessor.REGRESSION_HEAD);
		Configuration conf = new Configuration();

		Path output = new Path(".", "model.tfnn");
		TFNNModel.save(tfNNModel.getColumnIndexNameMap(), tfNNModel.getColumnTypeMap(),
				tfNNModel.getOneHotCategoryMap(), tfNNModel.getBasicNetwork(), FileSystem.getLocal(conf), output);
	}

	@Test
	public void testModel() throws IOException {
		String modelPath = "src/test/resources/tfmodel/model.tfnn";
		FileInputStream fi = null;
		TFNNModel tfNNModel;
		try {
			fi = new FileInputStream(modelPath);
			// long start = System.nanoTime();
			tfNNModel = TFNNModel.loadFromStream(fi);
		} finally {
			fi.close();
		}
		List<String> lines = FileUtils.readLines(new File("src/test/resources/tfmodel/Auto.csv"));

		if (lines.size() <= 1) {
			return;
		}
		String[] headers = CommonUtils.split(lines.get(0), ",");
		// score with format <String, String>
		for (int i = 1; i < lines.size(); i++) {
			Map<String, String> map = new HashMap<String, String>();
			Map<String, Object> mapObj = new HashMap<String, Object>();

			String[] data = CommonUtils.split(lines.get(i), ",");
			;
			// System.out.println("data len is " + data.length);
			if (data.length != headers.length) {
				System.out.println("One invalid input data");
				break;
			}
			for (int j = 0; j < headers.length; j++) {
				map.put(headers[j], data[j]);
				mapObj.put(headers[j], data[j]);
			}
			double[] scores = tfNNModel.compute(mapObj);
			System.out.println(Arrays.toString(scores));
		}

	}
}
