/**
 * Copyright [2012-2014] PayPal Software Foundation
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

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.Callable;

import ml.shifu.shifu.core.dtrain.dataset.BasicFloatMLDataPair;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.FloatFlatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.FloatMLDataPair;
import ml.shifu.shifu.core.dtrain.dataset.FloatMLDataSet;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.neural.error.ErrorFunction;
import org.encog.neural.flat.FlatNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link SubGradient} is copied from Encog framework. The reason is that we original Gradient don't pop up
 * {@link #gradients} outside. While we need gradients accumulated into {@link NNMaster} to update NN weights.
 */
public class SubGradient implements Callable<float[]> {

    protected static final Logger LOG = LoggerFactory.getLogger(SubGradient.class);

    /**
     * The network to train.
     */
    private FloatFlatNetwork network;

    /**
     * The error calculation method.
     */
    private ErrorCalculation errorCalculation = new SquaredErrorCalculation();

    /**
     * The actual values from the neural network.
     */
    private double[] actual;

    /**
     * The deltas for each layer.
     */
    private float[] layerDelta;

    /**
     * The neuron counts, per layer.
     */
    private int[] layerCounts;

    /**
     * The feed counts, per layer.
     */
    private int[] layerFeedCounts;

    /**
     * The layer indexes.
     */
    private int[] layerIndex;

    /**
     * The index to each layer's weights and thresholds.
     */
    private int[] weightIndex;

    /**
     * The output from each layer.
     */
    private float[] layerOutput;

    /**
     * The sums.
     */
    private float[] layerSums;

    /**
     * The gradients.
     */
    private float[] gradients;

    /**
     * The weights and thresholds.
     */
    private float[] weights;

    /**
     * The pair to use for training.
     */
    private FloatMLDataPair pair;

    /**
     * The training data.
     */
    private final FloatMLDataSet training;

    /**
     * The testing data, test data set here is used for training and testing cross over.
     */
    private FloatMLDataSet testing;

    /**
     * Whether to replace training and testing elements.
     */
    private final boolean isCrossOver;

    /**
     * Seed used to sample training and testing data set to choose which element is used for training
     */
    private long seed = System.currentTimeMillis();

    /**
     * error
     */
    private double error;

    /**
     * Derivative add constant. Used to combat flat spot.
     */
    private double[] flatSpot;

    /**
     * The error function to use.
     */
    private final ErrorFunction errorFunction;

    private final long trainLow;

    private final long trainHigh;

    private final long testLow;

    private final long testHigh;

    private ParallelGradient owner;

    private double[] doubleIdeal;

    /**
     * The dropout rate for each layer.
     */
    private float[] layerDropoutRates;

    /**
     * Used to generate randomness for dropout
     */
    protected Random dropoutRandomSource;

    /**
     * Current iteration
     */
    private int currentIteration;

    /**
     * If miniBatchRate set to 0.1d, {@link #batchs} is 10. It will run 10x iterations for one epochs.
     */
    private int batchs = 1;

    public SubGradient(final FloatFlatNetwork theNetwork, final FloatMLDataSet theTraining, long trainLow,
            long trainHigh, final FloatMLDataSet theTesting, long testLow, long testHigh, final double[] flatSpot,
            boolean isCrossOver, ParallelGradient owner, Random dropoutRandomSource, int batchs, int currentInteration) {
        this.network = theNetwork;
        this.training = theTraining;
        this.trainLow = trainLow;
        this.trainHigh = trainHigh;
        this.testing = theTesting;
        this.testLow = testLow;
        this.testHigh = testHigh;
        this.isCrossOver = isCrossOver;
        this.flatSpot = flatSpot;
        this.owner = owner;
        this.errorFunction = this.owner.createEFInstance();
        this.layerDropoutRates = theNetwork.getLayerDropoutRates();
        this.dropoutRandomSource = dropoutRandomSource;
        this.initNetworkParams();
        this.errorCalculation = this.owner.createECInstance();
        this.batchs = batchs;
        this.currentIteration = currentInteration;
    }

    private void initNetworkParams() {
        this.layerDelta = new float[this.network.getFloatLayerOutput().length];
        this.gradients = new float[this.network.getFloatWeights().length];
        this.actual = new double[this.network.getOutputCount()];

        this.weights = this.network.getFloatWeights();
        this.layerIndex = this.network.getLayerIndex();
        this.layerCounts = this.network.getLayerCounts();
        this.weightIndex = this.network.getWeightIndex();
        this.layerOutput = ((FloatFlatNetwork) this.getNetwork()).getFloatLayerOutput();
        this.layerSums = ((FloatFlatNetwork) this.getNetwork()).getFloatLayerSums();
        this.layerFeedCounts = this.network.getLayerFeedCounts();

        this.pair = BasicFloatMLDataPair.createPair(this.network.getInputCount(), getNetwork().getOutputCount());
    }

    /**
     * Process one training set element.
     * 
     * @param input
     *            The network input.
     * @param ideal
     *            The ideal values.
     * @param s
     *            The significance.
     */
    private void process(final float[] input, final float[] ideal, float s) {
        long start = System.currentTimeMillis();
        ((FloatFlatNetwork) this.getNetwork()).compute(input, this.actual);
        forwardTime += (System.currentTimeMillis() - start);

        // have to copy float ideal array to double array, since ideal array is small, it's ok to copy an array
        if(doubleIdeal == null) {
            doubleIdeal = new double[ideal.length];
        }
        for(int i = 0; i < doubleIdeal.length; i++) {
            doubleIdeal[i] = ideal[i];
        }

        start = System.currentTimeMillis();
        this.errorCalculation.updateError(this.actual, doubleIdeal, s);

        for(int i = 0; i < this.actual.length; i++) {
            float error = (float) (doubleIdeal[i] - actual[i]);
            this.layerDelta[i] = error;
        }

        // TODO this logic should be moved the ErrorFunction
        if(this.errorFunction instanceof LogErrorFunction) {
            for(int i = 0; i < this.actual.length; i++) {
                this.layerDelta[i] *= s;
            }
        } else {
            for(int i = 0; i < this.actual.length; i++) {
                this.layerDelta[i] = (float) (((this.getNetwork().getActivationFunctions()[0].derivativeFunction(
                        this.layerSums[i], this.layerOutput[i]) + this.flatSpot[0])) * (this.getLayerDelta()[i] * s));
            }
        }

        erroCompTime += (System.currentTimeMillis() - start);

        start = System.currentTimeMillis();
        int beginTraining = this.getNetwork().getBeginTraining();

        for(int i = beginTraining; i < this.getNetwork().getEndTraining(); i++) {
            processLevel(i);
        }
        backCompTime += (System.currentTimeMillis() - start);
    }

    /**
     * Process one level.
     * 
     * @param currentLevel
     *            The current level.
     */
    private void processLevel(final int currentLevel) {
        final int fromLayerIndex = this.layerIndex[currentLevel + 1];
        final int toLayerIndex = this.layerIndex[currentLevel];
        final int fromLayerSize = this.layerCounts[currentLevel + 1];
        final int toLayerSize = this.layerFeedCounts[currentLevel];
        double dropoutRate = 0;
        if(this.layerDropoutRates.length > currentLevel && this.layerDropoutRates[currentLevel] != 0) {
            dropoutRate = this.layerDropoutRates[currentLevel];
        }

        final int index = this.weightIndex[currentLevel];
        final ActivationFunction activation = this.getNetwork().getActivationFunctions()[currentLevel + 1];
        final double currentFlatSpot = this.flatSpot[currentLevel + 1];

        // handle weights
        int yi = fromLayerIndex;
        for(int y = 0; y < fromLayerSize; y++) {
            final float output = this.layerOutput[yi];
            float sum = 0;
            int wi = index + y;
            if(this.owner.isELM() || dropoutRate == 0d || dropoutRandomSource.nextDouble() > dropoutRate) {
                int xi = toLayerIndex;
                for(int x = 0; x < toLayerSize; x++) {
                    // if ELM and current Level is endTrainingIndex-1, from firstHidden to input, gradients set to 0 to
                    // skip update weights on input to first layer
                    if(this.owner.isELM() && currentLevel == (this.getNetwork().getEndTraining() - 1)) {
                        this.gradients[wi] = 0f;
                    } else {
                        this.gradients[wi] += output * this.getLayerDelta()[xi];
                    }
                    sum += this.weights[wi] * this.getLayerDelta()[xi];
                    wi += fromLayerSize;
                    xi++;
                }
                this.getLayerDelta()[yi] = sum
                        * (float) (activation.derivativeFunction(this.layerSums[yi], this.layerOutput[yi]) + currentFlatSpot);
            } else {
                this.getLayerDelta()[yi] = 0;
            }
            yi++;
        }
    }

    private long forwardTime = 0L;

    private long erroCompTime = 0L;

    private long backCompTime = 0L;

    /**
     * Perform the gradient calculation
     */
    @Override
    public final float[] call() {
        try {
            // reset errors and gradients firstly
            this.errorCalculation.reset();
            Arrays.fill(this.gradients, 0f);

            long start = this.trainLow;
            long end = this.trainHigh;

            if(this.batchs > 1) {
                long currentBatch = (currentIteration - 2) % this.batchs;
                long recordsInBatch = (this.trainHigh - this.trainLow + 1) / this.batchs;
                if(currentBatch == this.batchs - 1) {
                    start = this.trainLow + recordsInBatch * currentBatch;
                    end = this.trainHigh;
                } else {
                    start = this.trainLow + recordsInBatch * currentBatch;
                    end = this.trainLow + recordsInBatch * (currentBatch + 1) - 1;
                }
                LOG.debug(
                        "Thread name {}, trainLow {}, trainHigh {}, currentIteration {}, batchs {}, currentBatch {}, start {}, end {}",
                        Thread.currentThread().getName(), trainLow, trainHigh, currentIteration, batchs, currentBatch,
                        start, end);
            }

            for(long i = start; i <= end; i++) {
                synchronized(this.owner) {
                    if(this.isCrossOver) {
                        // 3:1 to select testing data set, tmp hard code, TODO fix hard code issue,extract such logic to
                        // a method
                        if((i + seed) % 4 < 3) {
                            this.training.getRecord(i, this.pair);
                        } else {
                            long testingSize = this.testing.getRecordCount();
                            // it's ok to take data from all testing set
                            if(i < testingSize) {
                                this.testing.getRecord(i, this.pair);
                            } else {
                                this.testing.getRecord(i % testingSize, this.pair);
                            }
                        }
                    } else {
                        this.training.getRecord(i, this.pair);
                    }
                }
                process(this.pair.getInputArray(), this.pair.getIdealArray(), pair.getSignificance());
            }
            this.error = this.errorCalculation.calculate();

            LOG.info("{}, forward time {}ms, error com time {}ms, back com time {}ms",
                    Thread.currentThread().getName(), forwardTime, erroCompTime, backCompTime);
        } catch (final Throwable ex) {
            throw new RuntimeException(ex);
        }
        return this.gradients;
    }

    /**
     * Calculate the error for this neural network. The error is calculated
     * using root-mean-square(RMS).
     * 
     * @param ec
     *            The error computation logic
     * @return The error percentage.
     */
    public final double calculateError(ErrorCalculation ec) {
        final double[] actual = new double[this.getNetwork().getOutputCount()];
        final FloatMLDataPair pair = BasicFloatMLDataPair.createPair(testing.getInputSize(), testing.getIdealSize());

        for(long i = testLow; i <= testHigh; i++) {
            synchronized(this.owner) {
                if(this.isCrossOver) {
                    // 3:1 to select testing data set, tmp hard code, TODO fix hard code issue
                    if((i + seed) % 4 < 3) {
                        this.testing.getRecord(i, pair);
                    } else {
                        long trainingSize = this.training.getRecordCount();
                        // it's ok to take data from all training set
                        if(i < trainingSize) {
                            this.training.getRecord(i, pair);
                        } else {
                            this.training.getRecord(i % trainingSize, pair);
                        }
                    }
                } else {
                    this.testing.getRecord(i, pair);
                }
            }
            ((FloatFlatNetwork) this.getNetwork()).compute(pair.getInputArray(), actual);
            // copy float idea array to double for api compatiability
            if(doubleIdeal == null) {
                doubleIdeal = new double[pair.getIdealArray().length];
            }
            for(int j = 0; j < doubleIdeal.length; j++) {
                doubleIdeal[j] = pair.getIdealArray()[j];
            }

            synchronized(ec) {
                ec.updateError(actual, doubleIdeal, pair.getSignificance());
            }
        }
        return -1;
    }

    public ErrorCalculation getErrorCalculation() {
        return errorCalculation;
    }

    /**
     * @return the gradients
     */
    public float[] getGradients() {
        return this.gradients;
    }

    /**
     * @return the error
     */
    public double getError() {
        return error;
    }

    /**
     * @return the weights
     */
    public float[] getWeights() {
        return weights;
    }

    /**
     * @param weights
     *            the weights to set
     */
    public void setWeights(float[] weights) {
        this.weights = weights;
        ((FloatFlatNetwork) (this.getNetwork())).setFloatWeights(weights);
    }

    public void setParams(BasicFloatNetwork network) {
        this.setNetwork((FloatFlatNetwork) network.getFlat());
        this.weights = ((FloatFlatNetwork) (network.getFlat())).getFloatWeights();
    }

    public FlatNetwork getNetwork() {
        return network;
    }

    public float[] getLayerDelta() {
        return layerDelta;
    }

    /**
     * @return the seed
     */
    public long getSeed() {
        return seed;
    }

    /**
     * @param seed
     *            the seed to set
     */
    public void setSeed(long seed) {
        this.seed = seed;
    }

    /**
     * @param network
     *            the network to set
     */
    public void setNetwork(FloatFlatNetwork network) {
        this.network = network;
        this.weights = this.network.getFloatWeights();
    }

}
