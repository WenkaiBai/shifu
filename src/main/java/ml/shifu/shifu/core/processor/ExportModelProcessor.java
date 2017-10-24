/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.processor;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ColumnStatsCalculator;
import ml.shifu.shifu.core.TreeModel;
import ml.shifu.shifu.core.binning.ColumnConfigDynamicBinning;
import ml.shifu.shifu.core.binning.obj.AbstractBinInfo;
import ml.shifu.shifu.core.binning.obj.CategoricalBinInfo;
import ml.shifu.shifu.core.binning.obj.NumericalBinInfo;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.FloatNeuralStructure;
import ml.shifu.shifu.core.dtrain.dt.BinaryDTSerializer;
import ml.shifu.shifu.core.dtrain.dt.TreeNode;
import ml.shifu.shifu.core.dtrain.nn.ActivationReLU;
import ml.shifu.shifu.core.dtrain.nn.BinaryNNSerializer;
import ml.shifu.shifu.core.dtrain.nn.TFNNModel;
import ml.shifu.shifu.core.pmml.PMMLTranslator;
import ml.shifu.shifu.core.pmml.PMMLUtils;
import ml.shifu.shifu.core.pmml.builder.PMMLConstructorFactory;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.HDFSUtils;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.MapUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.dmg.pmml.PMML;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.ml.BasicML;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.structure.NeuralStructure;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Output;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Session.Runner;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.framework.CollectionDef;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.NodeDef;

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.primitives.Booleans;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Floats;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import com.google.protobuf.InvalidProtocolBufferException;

/**
 * ExportModelProcessor class
 * 
 * @author zhanhu
 */
public class ExportModelProcessor extends BasicModelProcessor implements Processor {
    /**
     * log object
     */
    private final static Logger log = LoggerFactory.getLogger(ExportModelProcessor.class);

    public static final String PMML = "pmml";
    public static final String COLUMN_STATS = "columnstats";
    public static final String ONE_BAGGING_MODEL = "bagging";
    public static final String ONE_BAGGING_PMML_MODEL = "baggingpmml";
    public static final String WOE_MAPPING = "woemapping";
    public static final String TF_TO_SHIFU = "tf-2-shifu";

    public static final String IS_CONCISE = "IS_CONCISE";
    public static final String REQUEST_VARS = "REQUEST_VARS";
    public static final String EXPECTED_BIN_NUM = "EXPECTED_BIN_NUM";
    public static final String IV_KEEP_RATIO = "IV_KEEP_RATIO";
    public static final String MINIMUM_BIN_INST_CNT = "MINIMUM_BIN_INST_CNT";

    private String type;
    private Map<String, Object> params;

    public ExportModelProcessor(String type, Map<String, Object> params) {
        this.type = type;
        this.params = params;
    }

    public static final String REGRESSION_HEAD = "dnn/regression_head/predictions/scores";

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.shifu.core.processor.Processor#run()
     */
    @Override
    public int run() throws Exception {
        setUp(ModelStep.EXPORT);

        int status = 0;
        File pmmls = new File("pmmls");
        FileUtils.forceMkdir(pmmls);

        if(StringUtils.isBlank(type)) {
            type = PMML;
        }

        String modelsPath = pathFinder.getModelsPath(SourceType.LOCAL);
        if(type.equals(TF_TO_SHIFU)) {
            TFNNModel tfNNModel = createFromTfToEncog(new Path(modelsPath, "DNNRegressionAuto").toString(),
                    REGRESSION_HEAD);
            Configuration conf = new Configuration();

            Path output = new Path(modelsPath, "model.tfnn");
            TFNNModel.save(tfNNModel.getColumnIndexNameMap(), tfNNModel.getColumnTypeMap(),
                    tfNNModel.getOneHotCategoryMap(), tfNNModel.getBasicNetwork(), FileSystem.getLocal(conf), output);
        } else if(type.equalsIgnoreCase(ONE_BAGGING_MODEL)) {
            if(!"nn".equalsIgnoreCase(modelConfig.getAlgorithm())
                    && !CommonUtils.isTreeModel(modelConfig.getAlgorithm())) {
                log.warn("Currently one bagging model is only supported in NN/GBT/RF algorithm.");
            } else {
                List<BasicML> models = CommonUtils.loadBasicModels(modelsPath,
                        ALGORITHM.valueOf(modelConfig.getAlgorithm().toUpperCase()));
                if(models.size() < 1) {
                    log.warn("No model is found in {}.", modelsPath);
                } else {
                    log.info("Convert nn models into one binary bagging model.");
                    Configuration conf = new Configuration();
                    Path output = new Path(pathFinder.getBaggingModelPath(SourceType.LOCAL), "model.b"
                            + modelConfig.getAlgorithm());
                    if("nn".equalsIgnoreCase(modelConfig.getAlgorithm())) {
                        BinaryNNSerializer.save(modelConfig, columnConfigList, models, FileSystem.getLocal(conf),
                                output);
                    } else if(CommonUtils.isTreeModel(modelConfig.getAlgorithm())) {
                        List<List<TreeNode>> baggingTrees = new ArrayList<List<TreeNode>>();
                        for(int i = 0; i < models.size(); i++) {
                            TreeModel tm = (TreeModel) models.get(i);
                            // TreeModel only has one TreeNode instance although it is list inside
                            baggingTrees.add(tm.getIndependentTreeModel().getTrees().get(0));
                        }

                        int[] inputOutputIndex = DTrainUtils
                                .getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
                        // numerical + categorical = # of all input
                        int inputCount = inputOutputIndex[0] + inputOutputIndex[1];

                        BinaryDTSerializer.save(modelConfig, columnConfigList, baggingTrees, modelConfig.getParams()
                                .get("Loss").toString(), inputCount, FileSystem.getLocal(conf), output);
                    }
                    log.info("Please find one unified bagging model in local {}.", output);
                }
            }
        } else if(type.equalsIgnoreCase(PMML)) {
            // typical pmml generation
            List<BasicML> models = CommonUtils.loadBasicModels(modelsPath,
                    ALGORITHM.valueOf(modelConfig.getAlgorithm().toUpperCase()));

            PMMLTranslator translator = PMMLConstructorFactory.produce(modelConfig, columnConfigList, isConcise(),
                    false);

            for(int index = 0; index < models.size(); index++) {
                String path = "pmmls" + File.separator + modelConfig.getModelSetName() + Integer.toString(index)
                        + ".pmml";
                log.info("\t Start to generate " + path);
                PMML pmml = translator.build(Arrays.asList(new BasicML[] { models.get(index) }));
                PMMLUtils.savePMML(pmml, path);
            }
        } else if(type.equalsIgnoreCase(ONE_BAGGING_PMML_MODEL)) {
            // one unified bagging pmml generation
            log.info("Convert models into one bagging pmml model {} format", type);
            if(!"nn".equalsIgnoreCase(modelConfig.getAlgorithm())) {
                log.warn("Currently one bagging pmml model is only supported in NN algorithm.");
            } else {
                List<BasicML> models = CommonUtils.loadBasicModels(modelsPath,
                        ALGORITHM.valueOf(modelConfig.getAlgorithm().toUpperCase()));
                PMMLTranslator translator = PMMLConstructorFactory.produce(modelConfig, columnConfigList, isConcise(),
                        true);
                String path = "pmmls" + File.separator + modelConfig.getModelSetName() + ".pmml";
                log.info("\t Start to generate one unified model to: " + path);
                PMML pmml = translator.build(models);
                PMMLUtils.savePMML(pmml, path);
            }
        } else if(type.equalsIgnoreCase(COLUMN_STATS)) {
            saveColumnStatus();
        } else if(type.equalsIgnoreCase(WOE_MAPPING)) {
            List<ColumnConfig> exportCatColumns = new ArrayList<ColumnConfig>();
            List<String> catVariables = getRequestVars();
            for(ColumnConfig columnConfig: this.columnConfigList) {
                if(CollectionUtils.isEmpty(catVariables) || isRequestColumn(catVariables, columnConfig)) {
                    exportCatColumns.add(columnConfig);
                }
            }

            if(CollectionUtils.isNotEmpty(exportCatColumns)) {
                List<String> woeMappings = new ArrayList<String>();
                for(ColumnConfig columnConfig: exportCatColumns) {
                    String woeMapText = rebinAndExportWoeMapping(columnConfig);
                    woeMappings.add(woeMapText);
                }
                FileUtils.write(new File("woemapping.txt"), StringUtils.join(woeMappings, ",\n"));
            }
        } else {
            log.error("Unsupported output format - {}", type);
            status = -1;
        }

        clearUp(ModelStep.EXPORT);

        log.info("Done.");

        return status;
    }

    public static class TensorflowModel {

        MetaGraphDef metaGraphDef;

        GraphDef graphDef;

        Map<String, NodeDef> nodeMap;

        SavedModelBundle bundle;

        private Map<String, Map<?, ?>> tableMap = new LinkedHashMap<>();

        public TensorflowModel(SavedModelBundle model) {
            this.bundle = model;

            byte[] metaGraphDefBytes = model.metaGraphDef();

            try {
                metaGraphDef = MetaGraphDef.parseFrom(metaGraphDefBytes);
            } catch (InvalidProtocolBufferException e) {
                throw new RuntimeException(e);
            }

            GraphDef graphDef = metaGraphDef.getGraphDef();

            nodeMap = new LinkedHashMap<>();

            List<NodeDef> nodeDefs = graphDef.getNodeList();
            for(NodeDef nodeDef: nodeDefs) {
                nodeMap.put(nodeDef.getName(), nodeDef);
            }

            initializeTables();
        }

        private void initializeTables() {
            Collection<String> tableInitializerNames = Collections.emptyList();

            try {
                CollectionDef collectionDef = getCollectionDef("table_initializer");

                CollectionDef.NodeList nodeList = collectionDef.getNodeList();

                tableInitializerNames = nodeList.getValueList();
            } catch (IllegalArgumentException iae) {
                // Ignored
            }

            for(String tableInitializerName: tableInitializerNames) {
                NodeDef tableInitializer = getNodeDef(tableInitializerName);

                String name = tableInitializer.getInput(0);

                List<?> keys;
                List<?> values;

                try (Tensor tensor = run(tableInitializer.getInput(1))) {
                    keys = getValues(tensor);
                } // End try

                try (Tensor tensor = run(tableInitializer.getInput(2))) {
                    values = getValues(tensor);
                }

                Map<Object, Object> table = new LinkedHashMap<>();

                if(keys.size() != values.size()) {
                    throw new IllegalArgumentException();
                }

                for(int i = 0; i < keys.size(); i++) {
                    table.put(keys.get(i), values.get(i));
                }

                putTable(name, table);
            }
        }

        public void close() {
            SavedModelBundle bundle = getBundle();
            bundle.close();
        }

        public Tensor run(String name) {
            Session session = getSession();

            Runner runner = (session.runner()).fetch(name);

            List<Tensor> tensors = runner.run();

            return Iterables.getOnlyElement(tensors);
        }

        public Operation getOperation(String name) {
            Graph graph = getGraph();

            return graph.operation(name);
        }

        public NodeDef getNodeDef(String name) {
            Map<String, NodeDef> nodeMap = this.nodeMap;

            int colon = name.indexOf(':');

            NodeDef nodeDef = nodeMap.get(colon > -1 ? name.substring(0, colon) : name);
            if(nodeDef == null) {
                throw new IllegalArgumentException(name);
            }

            return nodeDef;
        }

        public CollectionDef getCollectionDef(String key) {
            MetaGraphDef metaGraphDef = this.metaGraphDef;

            Map<String, CollectionDef> collectionMap = metaGraphDef.getCollectionDefMap();

            CollectionDef collectionDef = collectionMap.get(key);
            if(collectionDef == null) {
                throw new IllegalArgumentException(key);
            }

            return collectionDef;
        }

        public NodeDef getOnlyInput(String name, String... ops) {
            Iterable<NodeDef> inputs = getInputs(name, ops);

            return Iterables.getOnlyElement(inputs);
        }

        public Iterable<NodeDef> getInputs(String name, String... ops) {
            NodeDef nodeDef = getNodeDef(name);

            Collection<Trail> trails = new LinkedHashSet<>();

            collectInputs(new ArrayDeque<NodeDef>(), nodeDef, new HashSet<>(Arrays.asList(ops)), trails);

            Function<Trail, NodeDef> function = new Function<Trail, NodeDef>() {
                @Override
                public NodeDef apply(Trail trail) {
                    return trail.getNodeDef();
                }
            };

            Collection<NodeDef> inputs = new LinkedHashSet<>();

            Iterables.addAll(inputs, Iterables.transform(trails, function));

            return inputs;
        }

        private void collectInputs(Deque<NodeDef> parentNodeDefs, NodeDef nodeDef, Set<String> ops,
                Collection<Trail> trails) {

            if(ops.contains(nodeDef.getOp())) {
                trails.add(new Trail(parentNodeDefs, nodeDef));
            }

            List<String> inputNames = nodeDef.getInputList();

            for(String inputName: inputNames) {
                NodeDef inputNodeDef = getNodeDef(inputName);

                parentNodeDefs.addFirst(inputNodeDef);

                collectInputs(parentNodeDefs, inputNodeDef, ops, trails);

                parentNodeDefs.removeFirst();
            }
        }

        static class Trail extends ArrayList<NodeDef> {

            private static final long serialVersionUID = -2772442671246820463L;

            Trail(Deque<NodeDef> parentNodeDefs, NodeDef nodeDef) {
                super(parentNodeDefs);
                add(nodeDef);
            }

            public NodeDef getNodeDef() {
                return get(size() - 1);
            }
        }

        public Map<?, ?> getTable(String name) {
            Map<?, ?> table = this.tableMap.get(name);

            if(table == null) {
                throw new IllegalArgumentException(name);
            }

            return table;
        }

        private void putTable(String name, Map<Object, Object> table) {
            this.tableMap.put(name, table);
        }

        public Session getSession() {
            SavedModelBundle bundle = getBundle();

            return bundle.session();
        }

        public Graph getGraph() {
            SavedModelBundle bundle = getBundle();

            return bundle.graph();
        }

        public SavedModelBundle getBundle() {
            return this.bundle;
        }

        //
        // private void setBundle(SavedModelBundle bundle){
        // this.bundle = bundle;
        // }
        //
        // public MetaGraphDef getMetaGraphDef(){
        // return this.metaGraphDef;
        // }
        //
        // private void setMetaGraphDef(MetaGraphDef metaGraphDef){
        // this.metaGraphDef = metaGraphDef;
        // }
        //
        // public Map<String, NodeDef> getNodeMap(){
        // return this.nodeMap;
        // }
        //
        // private void setNodeMap(Map<String, NodeDef> nodeMap){
        // this.nodeMap = nodeMap;
        // }

        static public List<?> getValues(Tensor tensor) {
            DataType dataType = tensor.dataType();

            switch(dataType) {
                case FLOAT:
                    return Floats.asList(toFloatArray(tensor));
                case DOUBLE:
                    return Doubles.asList(toDoubleArray(tensor));
                case INT32:
                    return Ints.asList(toIntArray(tensor));
                case INT64:
                    return Longs.asList(toLongArray(tensor));
                case STRING:
                    return Arrays.asList(toStringArray(tensor));
                case BOOL:
                    return Booleans.asList(toBooleanArray(tensor));
                default:
                    throw new IllegalArgumentException();
            }
        }

        static public float toFloatScalar(Tensor tensor) {

            try {
                return tensor.floatValue();
            } catch (Exception e) {
                float[] values = toFloatArray(tensor);

                if(values.length != 1) {
                    throw new IllegalArgumentException("Expected 1-element array, got " + Arrays.toString(values));
                }

                return values[0];
            }
        }

        static public float[] toFloatArray(Tensor tensor) {
            FloatBuffer floatBuffer = FloatBuffer.allocate(tensor.numElements());
            tensor.writeTo(floatBuffer);
            return floatBuffer.array();
        }

        static public double[] toDoubleArray(Tensor tensor) {
            DoubleBuffer doubleBuffer = DoubleBuffer.allocate(tensor.numElements());
            tensor.writeTo(doubleBuffer);
            return doubleBuffer.array();
        }

        static public int[] toIntArray(Tensor tensor) {
            IntBuffer intBuffer = IntBuffer.allocate(tensor.numElements());

            tensor.writeTo(intBuffer);

            return intBuffer.array();
        }

        static public long[] toLongArray(Tensor tensor) {
            LongBuffer longBuffer = LongBuffer.allocate(tensor.numElements());
            tensor.writeTo(longBuffer);
            return longBuffer.array();
        }

        static public String[] toStringArray(Tensor tensor) {
            ByteBuffer byteBuffer = ByteBuffer.allocate(tensor.numBytes());
            tensor.writeTo(byteBuffer);
            byteBuffer.position(tensor.numElements() * 8);
            String[] result = new String[tensor.numElements()];

            for(int i = 0; i < result.length; i++) {
                int length = byteBuffer.get();
                byte[] buffer = new byte[length];
                byteBuffer.get(buffer);
                result[i] = new String(buffer);
            }

            return result;
        }

        static public boolean[] toBooleanArray(Tensor tensor) {
            ByteBuffer byteBuffer = ByteBuffer.allocate(tensor.numElements());

            tensor.writeTo(byteBuffer);

            boolean[] result = new boolean[tensor.numElements()];

            for(int i = 0; i < result.length; i++) {
                result[i] = (byteBuffer.get(i) != 0);
            }

            return result;
        }

    }

    public static TFNNModel createFromTfToEncog(String tfModelPath, String head) throws InvalidProtocolBufferException {
        SavedModelBundle bundle = null;

        final BasicFloatNetwork network = new BasicFloatNetwork();
        try {
            bundle = SavedModelBundle.load(tfModelPath, "serve");
            TensorflowModel tfModel = new TensorflowModel(bundle);
            List<NodeDef> biasAdds = Lists.newArrayList(tfModel.getInputs(head, "BiasAdd"));

            biasAdds = Lists.reverse(biasAdds);

            SortedMap<Integer, String> columnIndexNameMap = new TreeMap<Integer, String>();

            Map<Integer, TFNNModel.DataType> columnTypeMap = new HashMap<Integer, TFNNModel.DataType>();

            Map<Integer, List<String>> oneHotCategoryMap = new HashMap<Integer, List<String>>();
            {
                NodeDef biasAdd = biasAdds.get(0);

                NodeDef matMul = tfModel.getNodeDef(biasAdd.getInput(0));
                if(!("MatMul").equals(matMul.getOp())) {
                    throw new IllegalArgumentException();
                }

                NodeDef concat = tfModel.getNodeDef(matMul.getInput(0));
                if(!("ConcatV2").equals(concat.getOp())) {
                    throw new IllegalArgumentException();
                }

                List<String> inputNames = concat.getInputList();
                for(int i = 0; i < inputNames.size() - 1; i++) {
                    String inputName = inputNames.get(i);

                    NodeDef term = tfModel.getNodeDef(inputName);

                    // "real_valued_column"
                    if(("Cast").equals(term.getOp()) || ("Placeholder").equals(term.getOp())) {
                        NodeDef placeholder = term;
                        if(("Cast").equals(placeholder.getOp())){
                            placeholder = tfModel.getNodeDef(placeholder.getInput(0));
                        }
                        columnIndexNameMap.put(i, placeholder.getName());
                        columnTypeMap.put(i, TFNNModel.DataType.DOUBLE);
                    } else
                    // "one_hot_column(sparse_column_with_keys)"
                    if(("Sum").equals(term.getOp())) {
                        NodeDef oneHot = tfModel.getOnlyInput(term.getInput(0), "OneHot");

                        NodeDef placeholder = tfModel.getOnlyInput(oneHot.getInput(0), "Placeholder");
                        NodeDef findTable = tfModel.getOnlyInput(oneHot.getInput(0), "LookupTableFind");

                        Map<?, ?> table = tfModel.getTable(findTable.getInput(0));

                        @SuppressWarnings({ "unchecked", "rawtypes" })
                        List<String> categories = (List) new ArrayList<>(table.keySet());

                        oneHotCategoryMap.put(i, categories);
                        columnIndexNameMap.put(i, placeholder.getName());
                        columnTypeMap.put(i, TFNNModel.DataType.ONEHOT);
                    } else {
                        throw new IllegalArgumentException(term.getName());
                    }

                }
            }

            int in = TFNNModel.getRealNNInput(columnTypeMap, oneHotCategoryMap);
            network.addLayer(new BasicLayer(new ActivationLinear(), true, in));

            for(int i = 0; i < biasAdds.size(); i++) {
                NodeDef biasAdd = biasAdds.get(i);

                NodeDef matMul = tfModel.getNodeDef(biasAdd.getInput(0));
                if(!("MatMul").equals(matMul.getOp())) {
                    throw new IllegalArgumentException();
                }

                int count;

                {
                    Operation operation = tfModel.getOperation(matMul.getName());

                    Output output = operation.output(0);

                    long[] shape = toArray(output.shape());
                    if(shape.length != 2 || shape[0] != -1) {
                        throw new IllegalArgumentException();
                    }

                    count = (int) shape[1];
                }

                if(i == biasAdds.size() - 1) {
                    network.addLayer(new BasicLayer(new ActivationLinear(), false, 1));
                } else {
                    network.addLayer(new BasicLayer(new ActivationReLU(), true, count));
                }
            }

            NeuralStructure structure = network.getStructure();
            if(network.getStructure() instanceof FloatNeuralStructure) {
                ((FloatNeuralStructure) structure).finalizeStruct();
            } else {
                structure.finalizeStructure();
            }

            int lastCount = in;
            for(int i = 0; i < biasAdds.size(); i++) {
                NodeDef biasAdd = biasAdds.get(i);

                NodeDef matMul = tfModel.getNodeDef(biasAdd.getInput(0));
                if(!("MatMul").equals(matMul.getOp())) {
                    throw new IllegalArgumentException();
                }

                int count;

                {
                    Operation operation = tfModel.getOperation(matMul.getName());

                    Output output = operation.output(0);

                    long[] shape = toArray(output.shape());
                    if(shape.length != 2 || shape[0] != -1) {
                        throw new IllegalArgumentException();
                    }

                    count = (int) shape[1];
                }

                NodeDef weights = tfModel.getOnlyInput(matMul.getInput(1), "VariableV2");

                float[] weightValues;

                try (Tensor tensor = tfModel.run(weights.getName())) {
                    weightValues = toFloatArray(tensor);
                }

                NodeDef bias = tfModel.getOnlyInput(biasAdd.getInput(1), "VariableV2");

                float[] biasValues;
                try (Tensor tensor = tfModel.run(bias.getName())) {
                    biasValues = toFloatArray(tensor);
                }

                for(int j = 0; j < count; j++) {
                    List<Float> entityWeights = getColumn(Floats.asList(weightValues), lastCount, count, j);
                    System.out.println(j + " " + entityWeights.size());
                    for(int k = 0; k < entityWeights.size(); k++) {
                        network.setWeight(i, k, j, (double) (entityWeights.get(k)));
                    }
                    // set bias
                    network.setWeight(i, entityWeights.size(), j, (double) (biasValues[j]));
                }

                lastCount = count;
            }

            return new TFNNModel(network, columnIndexNameMap, columnTypeMap, oneHotCategoryMap);
        } finally {
            if(bundle != null) {
                bundle.close();
            }
        }

    }

    static public <E> List<E> getColumn(List<E> values, int rows, int columns, int index) {
        // validateSize(values, rows, columns);

        List<E> result = new ArrayList<>(rows);

        for(int row = 0; row < rows; row++) {
            E value = values.get((row * columns) + index);

            result.add(value);
        }

        return result;
    }

    static public float[] toFloatArray(Tensor tensor) {
        FloatBuffer floatBuffer = FloatBuffer.allocate(tensor.numElements());

        tensor.writeTo(floatBuffer);

        return floatBuffer.array();
    }

    static public long[] toArray(Shape shape) {
        int length = shape.numDimensions();

        if(length < 0) {
            return null;
        }

        long[] result = new long[length];

        for(int i = 0; i < length; i++) {
            result[i] = shape.size(i);
        }

        return result;
    }

    // public ContinuousFeature createContinuousFeature(SavedModel savedModel, NodeDef placeholder){
    // NodeDef cast = null;
    //
    // if(("Cast").equals(placeholder.getOp())){
    // cast = placeholder;
    // placeholder = savedModel.getNodeDef(placeholder.getInput(0));
    // }
    //
    // DataField dataField = ensureContinuousDataField(savedModel, placeholder);
    //
    // ContinuousFeature result = new ContinuousFeature(this, dataField);
    //
    // if(cast != null){
    // Operation operation = savedModel.getOperation(cast.getName());
    //
    // Output output = operation.output(0);
    //
    // result = result.toContinuousFeature(TypeUtil.getDataType(output));
    // }
    //
    // return result;
    // }

    // public List<BinaryFeature> createBinaryFeatures(SavedModel savedModel, NodeDef placeholder, List<String>
    // categories) {
    // DataField dataField = ensureCategoricalDataField(savedModel, placeholder, categories);
    //
    // List<BinaryFeature> result = new ArrayList<>();
    //
    // for(String category: categories) {
    // BinaryFeature binaryFeature = new BinaryFeature(this, dataField, category);
    //
    // result.add(binaryFeature);
    // }
    //
    // return result;
    // }

    private String rebinAndExportWoeMapping(ColumnConfig columnConfig) throws IOException {
        int expectBinNum = getExpectBinNum();
        double ivKeepRatio = getIvKeepRatio();
        long minimumInstCnt = getMinimumInstCnt();

        ColumnConfigDynamicBinning columnConfigDynamicBinning = new ColumnConfigDynamicBinning(columnConfig,
                expectBinNum, ivKeepRatio, minimumInstCnt);

        List<AbstractBinInfo> binInfos = columnConfigDynamicBinning.run();

        long[] binCountNeg = new long[binInfos.size() + 1];
        long[] binCountPos = new long[binInfos.size() + 1];
        for(int i = 0; i < binInfos.size(); i++) {
            AbstractBinInfo binInfo = binInfos.get(i);
            binCountNeg[i] = binInfo.getNegativeCnt();
            binCountPos[i] = binInfo.getPositiveCnt();
        }
        binCountNeg[binCountNeg.length - 1] = columnConfig.getBinCountNeg().get(
                columnConfig.getBinCountNeg().size() - 1);
        binCountPos[binCountPos.length - 1] = columnConfig.getBinCountPos().get(
                columnConfig.getBinCountPos().size() - 1);
        ColumnStatsCalculator.ColumnMetrics columnMetrics = ColumnStatsCalculator.calculateColumnMetrics(binCountNeg,
                binCountPos);

        System.out.println(columnConfig.getColumnName() + ":");
        for(int i = 0; i < binInfos.size(); i++) {
            if(columnConfig.isCategorical()) {
                CategoricalBinInfo binInfo = (CategoricalBinInfo) binInfos.get(i);
                System.out.println("\t" + binInfo.getValues() + " | posCount:" + binInfo.getPositiveCnt()
                        + " | negCount:" + binInfo.getNegativeCnt() + " | posRate:" + binInfo.getPositiveRate()
                        + " | woe:" + columnMetrics.getBinningWoe().get(i));
            } else {
                NumericalBinInfo binInfo = (NumericalBinInfo) binInfos.get(i);
                System.out.println("\t[" + binInfo.getLeftThreshold() + ", " + binInfo.getRightThreshold() + ")"
                        + " | posCount:" + binInfo.getPositiveCnt() + " | negCount:" + binInfo.getNegativeCnt()
                        + " | posRate:" + binInfo.getPositiveRate() + " | woe:" + columnMetrics.getBinningWoe().get(i));
            }
        }
        System.out.println("\t" + columnConfig.getColumnName() + " IV:" + columnMetrics.getIv());
        System.out.println("\t" + columnConfig.getColumnName() + " KS:" + columnMetrics.getKs());
        System.out.println("\t" + columnConfig.getColumnName() + " WOE:" + columnMetrics.getWoe());
        return generateWoeMapping(columnConfig, binInfos, columnMetrics);
    }

    private String generateWoeMapping(ColumnConfig columnConfig, List<AbstractBinInfo> binInfos,
            ColumnStatsCalculator.ColumnMetrics columnMetrics) {
        if(columnConfig.isCategorical()) {
            return generateCategoricalWoeMapping(columnConfig, binInfos, columnMetrics);
        } else {
            return generateNumericalWoeMapping(columnConfig, binInfos, columnMetrics);
        }
    }

    private String generateNumericalWoeMapping(ColumnConfig columnConfig, List<AbstractBinInfo> numericalBinInfos,
            ColumnStatsCalculator.ColumnMetrics columnMetrics) {
        StringBuffer buffer = new StringBuffer();
        buffer.append("( case \n");
        buffer.append("\twhen " + columnConfig.getColumnName() + " = . then "
                + columnMetrics.getBinningWoe().get(columnMetrics.getBinningWoe().size() - 1) + "\n");
        for(int i = 0; i < numericalBinInfos.size(); i++) {
            NumericalBinInfo binInfo = (NumericalBinInfo) numericalBinInfos.get(i);
            buffer.append("\twhen (");
            if(!Double.isInfinite(binInfo.getLeftThreshold())) {
                buffer.append(binInfo.getLeftThreshold() + " <= ");
            }
            buffer.append(columnConfig.getColumnName());
            if(!Double.isInfinite(binInfo.getRightThreshold())) {
                buffer.append(" < " + binInfo.getRightThreshold());
            }
            buffer.append(") then " + columnMetrics.getBinningWoe().get(i) + "\n");
        }
        buffer.append("  end ) as " + columnConfig.getColumnName() + "_" + numericalBinInfos.size());
        return buffer.toString();
    }

    private String generateCategoricalWoeMapping(ColumnConfig columnConfig, List<AbstractBinInfo> categoricalBinInfos,
            ColumnStatsCalculator.ColumnMetrics columnMetrics) {
        StringBuffer buffer = new StringBuffer();
        buffer.append("( case \n");
        for(int i = 0; i < categoricalBinInfos.size(); i++) {
            CategoricalBinInfo binInfo = (CategoricalBinInfo) categoricalBinInfos.get(i);
            List<String> values = new ArrayList<String>();
            for(String cval: binInfo.getValues()) {
                List<String> subCvals = CommonUtils.flattenCatValGrp(cval);
                for(String subCval: subCvals) {
                    values.add("'" + subCval + "'");
                }
            }
            buffer.append("\twhen " + columnConfig.getColumnName() + " in (" + StringUtils.join(values, ',')
                    + ") then " + columnMetrics.getBinningWoe().get(i) + "\n");
        }
        buffer.append("\telse " + columnMetrics.getBinningWoe().get(columnMetrics.getBinningWoe().size() - 1) + "\n");
        buffer.append("  end ) as " + columnConfig.getColumnName() + "_" + categoricalBinInfos.size());
        return buffer.toString();
    }

    private void saveColumnStatus() throws IOException {
        Path localColumnStatsPath = new Path(pathFinder.getLocalColumnStatsPath());
        log.info("Saving ColumnStatus to local file system: {}.", localColumnStatsPath);
        if(HDFSUtils.getLocalFS().exists(localColumnStatsPath)) {
            HDFSUtils.getLocalFS().delete(localColumnStatsPath, true);
        }

        BufferedWriter writer = null;
        try {
            writer = ShifuFileUtils.getWriter(localColumnStatsPath.toString(), SourceType.LOCAL);
            writer.write("dataSet,columnFlag,columnName,columnNum,iv,ks,max,mean,median,min,missingCount,"
                    + "missingPercentage,stdDev,totalCount,distinctCount,weightedIv,weightedKs,weightedWoe,woe,"
                    + "skewness,kurtosis,columnType,finalSelect,psi,unitstats,version\n");
            StringBuilder builder = new StringBuilder(500);
            for(ColumnConfig columnConfig: columnConfigList) {
                builder.setLength(0);
                builder.append(modelConfig.getBasic().getName()).append(',');
                builder.append(columnConfig.getColumnFlag()).append(',');
                builder.append(columnConfig.getColumnName()).append(',');
                builder.append(columnConfig.getColumnNum()).append(',');
                builder.append(columnConfig.getIv()).append(',');
                builder.append(columnConfig.getKs()).append(',');
                builder.append(columnConfig.getColumnStats().getMax()).append(',');
                builder.append(columnConfig.getColumnStats().getMean()).append(',');
                builder.append(columnConfig.getColumnStats().getMedian()).append(',');
                builder.append(columnConfig.getColumnStats().getMin()).append(',');
                builder.append(columnConfig.getColumnStats().getMissingCount()).append(',');
                builder.append(columnConfig.getColumnStats().getMissingPercentage()).append(',');
                builder.append(columnConfig.getColumnStats().getStdDev()).append(',');
                builder.append(columnConfig.getColumnStats().getTotalCount()).append(',');
                builder.append(columnConfig.getColumnStats().getDistinctCount()).append(',');
                builder.append(columnConfig.getColumnStats().getWeightedIv()).append(',');
                builder.append(columnConfig.getColumnStats().getWeightedKs()).append(',');
                builder.append(columnConfig.getColumnStats().getWeightedWoe()).append(',');
                builder.append(columnConfig.getColumnStats().getWoe()).append(',');
                builder.append(columnConfig.getColumnStats().getSkewness()).append(',');
                builder.append(columnConfig.getColumnStats().getKurtosis()).append(',');
                builder.append(columnConfig.getColumnType()).append(',');
                builder.append(columnConfig.isFinalSelect()).append(',');
                builder.append(columnConfig.getPSI()).append(',');
                builder.append(StringUtils.join(columnConfig.getUnitStats(), '|')).append(',');
                builder.append(modelConfig.getBasic().getVersion()).append("\n");
                writer.write(builder.toString());
            }
        } finally {
            writer.close();
        }
    }

    private boolean isConcise() {
        if(MapUtils.isNotEmpty(this.params) && this.params.get(IS_CONCISE) instanceof Boolean) {
            return (Boolean) this.params.get(IS_CONCISE);
        }
        return false;
    }

    private List<String> getRequestVars() {
        if(MapUtils.isNotEmpty(this.params) && this.params.get(REQUEST_VARS) instanceof String) {
            String requestVars = (String) this.params.get(REQUEST_VARS);
            if(StringUtils.isNotBlank(requestVars)) {
                return Arrays.asList(requestVars.split(","));
            }
        }
        return null;
    }

    private int getExpectBinNum() {
        if(MapUtils.isNotEmpty(this.params) && this.params.get(EXPECTED_BIN_NUM) instanceof String) {
            String expectBinNum = (String) this.params.get(EXPECTED_BIN_NUM);
            try {
                return Integer.parseInt(expectBinNum);
            } catch (Exception e) {
                log.warn("Invalid expect bin num {}. Ignore it...", expectBinNum);
            }
        }
        return 0;
    }

    public double getIvKeepRatio() {
        if(MapUtils.isNotEmpty(this.params) && this.params.get(IV_KEEP_RATIO) instanceof String) {
            String ivKeepRatio = (String) this.params.get(IV_KEEP_RATIO);
            try {
                return Double.parseDouble(ivKeepRatio);
            } catch (Exception e) {
                log.warn("Invalid IV Keep ratio {}. Ignore it...", ivKeepRatio);
            }
        }
        return 1.0;
    }

    public long getMinimumInstCnt() {
        if(MapUtils.isNotEmpty(this.params) && this.params.get(MINIMUM_BIN_INST_CNT) instanceof String) {
            String minimumBinInstCnt = (String) this.params.get(MINIMUM_BIN_INST_CNT);
            try {
                return Long.parseLong(minimumBinInstCnt);
            } catch (Exception e) {
                log.warn("Invalid minimum bin instance count {}. Ignore it...", minimumBinInstCnt);
            }
        }
        return 0;
    }
}
