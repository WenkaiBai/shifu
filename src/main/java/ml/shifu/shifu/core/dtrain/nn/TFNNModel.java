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
package ml.shifu.shifu.core.dtrain.nn;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.PersistBasicFloatNetwork;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.encog.ml.BasicML;
import org.encog.ml.data.basic.BasicMLData;

/**
 * TODO
 * 
 * @author pengzhang
 */
public class TFNNModel {

    /**
     * Encog based neural network instance which is used to compute nn score
     */
    private final BasicFloatNetwork basicNetwork;

    public static enum DataType {
        DOUBLE(0), ONEHOT(1);

        private final int type;

        private DataType(int type) {
            this.type = type;
        }

        /**
         * @return the type
         */
        public int getType() {
            return type;
        }

        public static DataType findDataType(int type) {
            for(DataType dataType: values()) {
                if(type == dataType.type) {
                    return dataType;
                }
            }
            return null;
        }
    }

    private final SortedMap<Integer, String> columnIndexNameMap;

    private final Map<Integer, DataType> columnTypeMap;

    private final Map<Integer, List<String>> oneHotCategoryMap;

    /**
     * @return the basicNetwork
     */
    public BasicFloatNetwork getBasicNetwork() {
        return basicNetwork;
    }

    /**
     * @return the columnIndexNameMap
     */
    public SortedMap<Integer, String> getColumnIndexNameMap() {
        return columnIndexNameMap;
    }

    /**
     * @return the columnTypeMap
     */
    public Map<Integer, DataType> getColumnTypeMap() {
        return columnTypeMap;
    }

    /**
     * @return the oneHotCategoryMap
     */
    public Map<Integer, List<String>> getOneHotCategoryMap() {
        return oneHotCategoryMap;
    }

    public TFNNModel(BasicFloatNetwork basicNetwork, SortedMap<Integer, String> columnIndexNameMap,
            Map<Integer, DataType> columnTypeMap, Map<Integer, List<String>> oneHotCategoryMap) {
        this.basicNetwork = basicNetwork;
        this.columnIndexNameMap = columnIndexNameMap;
        this.columnTypeMap = columnTypeMap;
        this.oneHotCategoryMap = oneHotCategoryMap;
    }

    public double[] compute(double[] data) {
        if(this.basicNetwork == null) {
            throw new IllegalStateException("no models inside");
        }

        return this.basicNetwork.compute(new BasicMLData(data)).getData();
    }

    public double[] compute(Map<String, Object> dataMap) {
        return compute(convertDataMapToDoubleArray(dataMap));
    }

    public static int getRealNNInput(Map<Integer, DataType> columnTypeMap, Map<Integer, List<String>> oneHotCategoryMap) {
        int inputCnt = 0;
        for(Entry<Integer, DataType> entry: columnTypeMap.entrySet()) {
            if(entry.getValue() == DataType.DOUBLE) {
                inputCnt += 1;
            } else if(entry.getValue() == DataType.ONEHOT) {
                inputCnt += oneHotCategoryMap.get(entry.getKey()).size();
            }
        }
        return inputCnt;
    }

    private double[] convertDataMapToDoubleArray(Map<String, Object> dataMap) {
        int inputCnt = 0;
        for(Entry<Integer, DataType> entry: columnTypeMap.entrySet()) {
            if(entry.getValue() == DataType.DOUBLE) {
                inputCnt += 1;
            } else if(entry.getValue() == DataType.ONEHOT) {
                inputCnt += oneHotCategoryMap.get(entry.getKey()).size();
            }
        }

        double[] data = new double[inputCnt];

        int index = 0;
        for(Entry<Integer, String> entry: columnIndexNameMap.entrySet()) {
            if(columnTypeMap.get(entry.getKey()) == DataType.DOUBLE) {
                try {
                    data[index] = Double.parseDouble(dataMap.get(entry.getValue()).toString());
                } catch (Exception e) {
                    e.printStackTrace();
                }
                index += 1;
            } else if(columnTypeMap.get(entry.getKey()) == DataType.ONEHOT) {
                List<String> categories = oneHotCategoryMap.get(entry.getKey());
                int cateIndex = categories.indexOf(dataMap.get(entry.getValue()).toString());
                for(int i = 0; i < categories.size(); i++) {
                    data[index + i] = (i == cateIndex ? 1d : 0d);
                }
                index += categories.size();
            }
        }
        return data;
    }

    public static void save(SortedMap<Integer, String> columnNameIndexMap, Map<Integer, DataType> columnTypeMap,
            Map<Integer, List<String>> oneHotCategoryMap, BasicML basicNetwork, FileSystem fs, Path output)
            throws IOException {
        DataOutputStream fos = null;
        try {
            fos = new DataOutputStream(new GZIPOutputStream(fs.create(output)));

            // version
            fos.writeInt(CommonConstants.NN_FORMAT_VERSION);

            // write column index mapping
            fos.writeInt(columnNameIndexMap.size());
            for(Entry<Integer, String> entry: columnNameIndexMap.entrySet()) {
                fos.writeInt(entry.getKey());
                fos.writeUTF(entry.getValue());
            }

            // write column type mapping
            fos.writeInt(columnTypeMap.size());
            for(Entry<Integer, DataType> entry: columnTypeMap.entrySet()) {
                fos.writeInt(entry.getKey());
                fos.writeInt(entry.getValue().getType());
            }

            // write column category list mapping
            fos.writeInt(oneHotCategoryMap.size());
            for(Entry<Integer, List<String>> entry: oneHotCategoryMap.entrySet()) {
                fos.writeInt(entry.getKey());
                List<String> categories = entry.getValue();
                fos.writeInt(categories.size());
                for(int i = 0; i < categories.size(); i++) {
                    fos.writeUTF(categories.get(i));
                }
            }

            // persist network
            new PersistBasicFloatNetwork().saveNetwork(fos, (BasicFloatNetwork) basicNetwork);
        } finally {
            IOUtils.closeStream(fos);
        }
    }

    public static TFNNModel loadFromStream(InputStream input) throws IOException {
        DataInputStream dis = null;
        // check if gzip or not
        try {
            byte[] header = new byte[2];
            BufferedInputStream bis = new BufferedInputStream(input);
            bis.mark(2);
            int result = bis.read(header);
            bis.reset();
            int ss = (header[0] & 0xff) | ((header[1] & 0xff) << 8);
            if(result != -1 && ss == GZIPInputStream.GZIP_MAGIC) {
                dis = new DataInputStream(new GZIPInputStream(bis));
            } else {
                dis = new DataInputStream(bis);
            }
        } catch (java.io.IOException e) {
            dis = new DataInputStream(input);
        }

        @SuppressWarnings("unused")
        int version = dis.readInt();

        // for index name mapping
        SortedMap<Integer, String> indexNameMap = new TreeMap<Integer, String>();
        int size = dis.readInt();
        for(int i = 0; i < size; i++) {
            int key = dis.readInt();
            String value = dis.readUTF();
            indexNameMap.put(key, value);
        }

        // for index type mapping
        Map<Integer, DataType> indexTypeMap = new HashMap<Integer, DataType>();
        size = dis.readInt();
        for(int i = 0; i < size; i++) {
            int key = dis.readInt();
            indexTypeMap.put(key, DataType.findDataType(dis.readInt()));
        }

        // for categorical features
        Map<Integer, List<String>> cateColumnIndexNames = new HashMap<Integer, List<String>>();
        size = dis.readInt();
        for(int i = 0; i < size; i++) {
            int key = dis.readInt();
            int cateSize = dis.readInt();
            List<String> cates = new ArrayList<String>();
            for(int j = 0; j < cateSize; j++) {
                cates.add(dis.readUTF());
            }
            cateColumnIndexNames.put(key, cates);
        }

        return new TFNNModel(new PersistBasicFloatNetwork().readNetwork(dis), indexNameMap, indexTypeMap,
                cateColumnIndexNames);
    }

}
