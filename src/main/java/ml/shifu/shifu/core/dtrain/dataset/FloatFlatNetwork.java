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
package ml.shifu.shifu.core.dtrain.dataset;

import java.io.Serializable;

import ml.shifu.shifu.core.dtrain.nn.ActivationReLU;
import ml.shifu.shifu.core.dtrain.nn.BasicDropoutLayer;

import org.encog.EncogError;
import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationLOG;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSIN;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.mathutil.BoundMath;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.neural.flat.FlatLayer;
import org.encog.neural.flat.FlatNetwork;
import org.encog.util.EngineArray;

/**
 * To solve float input and output types.
 */
public class FloatFlatNetwork extends FlatNetwork implements Cloneable, Serializable {

    private static final long serialVersionUID = -7208969306860840672L;

    /**
     * The dropout rate for each layer.
     */
    private float[] floatLayerDropoutRates;

    /**
     * The outputs from each of the neurons.
     */
    private float[] floatLayerOutput;

    /**
     * The sum of the layer, before the activation function is applied, producing the layerOutput.
     */
    private float[] floatLayerSums;

    /**
     * The weights for a neural network.
     */
    private float[] floatWeights;

    public FloatFlatNetwork() {
        this.floatLayerDropoutRates = new float[0];
    }

    public FloatFlatNetwork(final FlatLayer[] layers) {
        this(layers, true);
    }

    public FloatFlatNetwork(final FlatLayer[] layers, boolean dropout) {
        initNetwork(layers, dropout);
    }

    protected void initNetwork(FlatLayer[] layers, boolean dropout) {
        super.init(layers);

        int neuronCount = 0;
        int weightCount = super.getWeights().length;
        for(int i = layers.length - 1; i >= 0; i--) {
            neuronCount += layers[i].getTotalCount();
        }

        this.floatWeights = new float[weightCount];
        this.floatLayerOutput = new float[neuronCount];
        this.floatLayerSums = new float[neuronCount];

        final int layerCount = layers.length;
        if(dropout) {
            this.floatLayerDropoutRates = new float[layerCount];
        } else {
            this.floatLayerDropoutRates = new float[0];
        }

        int index = 0;
        for(int i = layers.length - 1; i >= 0; i--) {
            final FlatLayer layer = layers[i];
            if(dropout && layer instanceof BasicDropoutLayer) {
                this.getLayerDropoutRates()[index] = (float) ((BasicDropoutLayer) layer).getDropout();
            }
            index += 1;
        }

        clearFloatContext();

        super.setLayerOutput(null);
        super.setLayerSums(null);
        super.setWeights(null);
    }

    /**
     * Clear any context neurons.
     */
    public void clearFloatContext() {
        int index = 0;
        for(int i = 0; i < super.getLayerIndex().length; i++) {

            final boolean hasBias = (super.getLayerContextCount()[i] + super.getLayerFeedCounts()[i]) != super
                    .getLayerCounts()[i];

            // fill in regular neurons
            for(int j = 0; j < this.getLayerFeedCounts()[i]; j++) {
                this.floatLayerOutput[index++] = 0;
            }

            // fill in the bias
            if(hasBias) {
                this.floatLayerOutput[index++] = (float) super.getBiasActivation()[i];
            }

            // fill in context
            for(int j = 0; j < this.getLayerContextCount()[i]; j++) {
                this.floatLayerOutput[index++] = 0;
            }
        }
    }

    public void compute(float[] input, double[] output) {
        final int sourceIndex = this.floatLayerOutput.length - getLayerCounts()[getLayerCounts().length - 1];

        for(int i = 0; i < getInputCount(); i++) {
            this.floatLayerOutput[i + sourceIndex] = input[i];
        }

        for(int i = this.getLayerIndex().length - 1; i > 0; i--) {
            computeLayer(i);
        }

        // update context values
        final int offset = getContextTargetOffset()[0];

        for(int x = 0; x < getContextTargetSize()[0]; x++) {
            this.floatLayerOutput[offset + x] = this.floatLayerOutput[x];
        }

        for(int i = 0; i < this.getOutputCount(); i++) {
            output[i] = this.floatLayerOutput[i];
        }
    }

    @Override
    public void compute(final double[] input, final double[] output) {
        final int sourceIndex = this.floatLayerOutput.length - getLayerCounts()[getLayerCounts().length - 1];

        for(int i = 0; i < getInputCount(); i++) {
            this.floatLayerOutput[i + sourceIndex] = (float) input[i];
        }

        for(int i = this.getLayerIndex().length - 1; i > 0; i--) {
            computeLayer(i);
        }

        // update context values
        final int offset = getContextTargetOffset()[0];

        for(int x = 0; x < getContextTargetSize()[0]; x++) {
            this.floatLayerOutput[offset + x] = this.floatLayerOutput[x];
        }

        for(int i = 0; i < this.getOutputCount(); i++) {
            output[i] = this.floatLayerOutput[i];
        }
    }

    public void compute(float[] input, float[] output) {
        final int sourceIndex = this.floatLayerOutput.length - getLayerCounts()[getLayerCounts().length - 1];

        for(int i = 0; i < getInputCount(); i++) {
            this.floatLayerOutput[i + sourceIndex] = input[i];
        }

        for(int i = getLayerIndex().length - 1; i > 0; i--) {
            computeLayer(i);
        }

        // update context values
        final int offset = getContextTargetOffset()[0];

        for(int x = 0; x < getContextTargetSize()[0]; x++) {
            this.floatLayerOutput[offset + x] = this.floatLayerOutput[x];
        }

        // copy to float output array
        for(int i = 0; i < getOutputCount(); i++) {
            output[i] = (float) this.floatLayerOutput[i];
        }

        System.arraycopy(this.floatLayerOutput, 0, output, 0, super.getOutputCount());
    }

    @Override
    protected void computeLayer(final int currentLayer) {
        final int inputIndex = super.getLayerIndex()[currentLayer];
        final int outputIndex = super.getLayerIndex()[currentLayer - 1];
        final int inputSize = super.getLayerCounts()[currentLayer];
        final int outputSize = super.getLayerFeedCounts()[currentLayer - 1];
        final float dropoutRate;
        if(this.getLayerDropoutRates().length > currentLayer - 1) {
            dropoutRate = this.getLayerDropoutRates()[currentLayer - 1];
        } else {
            dropoutRate = 0f;
        }
        boolean dropoutEnabled = (Double.compare(dropoutRate, 0d) != 0);

        int index = super.getWeightIndex()[currentLayer - 1];

        final int limitX = outputIndex + outputSize;
        final int limitY = inputIndex + inputSize;

        // wrapper computation in if condition to save computation
        if(dropoutEnabled) {
            // weight values
            float nonDropoutRate = (1f - dropoutRate);
            for(int x = outputIndex; x < limitX; x++) {
                float sum = 0;
                for(int y = inputIndex; y < limitY; y++) {
                    sum += this.floatWeights[index++] * this.floatLayerOutput[y] * nonDropoutRate;
                }
                this.floatLayerSums[x] = sum;
                this.floatLayerOutput[x] = sum;
            }
        } else {
            // weight values
            for(int x = outputIndex; x < limitX; x++) {
                float sum = 0;
                for(int y = inputIndex; y < limitY; y++) {
                    sum += this.floatWeights[index++] * this.floatLayerOutput[y];
                }
                this.floatLayerSums[x] = sum;
                this.floatLayerOutput[x] = sum;
            }
        }

        // do activation for float type here
        ActivationFunction af = super.getActivationFunctions()[currentLayer - 1];
        if(af instanceof ActivationSigmoid) {
            for(int i = outputIndex; i < outputIndex + outputSize; i++) {
                this.floatLayerOutput[i] = (float) (1.0 / (1.0 + BoundMath.exp(-1 * this.floatLayerOutput[i])));
            }
        } else if(af instanceof ActivationTANH) {
            for(int i = outputIndex; i < outputIndex + outputSize; i++) {
                this.floatLayerOutput[i] = (float) Math.tanh(this.floatLayerOutput[i]);
            }
        } else if(af instanceof ActivationReLU) {
            for(int i = outputIndex; i < outputIndex + outputSize; i++) {
                float output = this.floatLayerOutput[i];
                if(output <= 0f) {
                    this.floatLayerOutput[i] = 0f;
                }
            }
        } else if(af instanceof ActivationSIN) {
            for(int i = outputIndex; i < outputIndex + outputSize; i++) {
                this.floatLayerOutput[i] = (float) BoundMath.sin(this.floatLayerOutput[i]);
            }
        } else if(af instanceof ActivationLOG) {
            for(int i = outputIndex; i < outputIndex + outputSize; i++) {
                if(this.floatLayerOutput[i] >= 0) {
                    this.floatLayerOutput[i] = (float) BoundMath.log(1 + this.floatLayerOutput[i]);
                } else {
                    this.floatLayerOutput[i] = (float) -BoundMath.log(1 - this.floatLayerOutput[i]);
                }
            }
        } else if(af instanceof ActivationLinear) {
            // activation for linear is doing nothing.
        }

        // update context values
        final int offset = super.getContextTargetOffset()[currentLayer];

        for(int x = 0; x < super.getContextTargetSize()[currentLayer]; x++) {
            this.floatLayerOutput[offset + x] = this.floatLayerOutput[outputIndex + x];
        }
    }

    /**
     * Clone the network.
     * 
     * @return A clone of the network.
     */
    public FloatFlatNetwork clone() {
        final FloatFlatNetwork result = new FloatFlatNetwork();
        this.cloneFloatFlatNetwork(result);
        return result;
    }

    public final void cloneFloatFlatNetwork(final FloatFlatNetwork result) {
        result.setInputCount(super.getInputCount());
        result.setLayerCounts(EngineArray.arrayCopy(super.getLayerCounts()));
        result.setLayerIndex(EngineArray.arrayCopy(super.getLayerIndex()));
        result.setFloatLayerOutput(arrayCopy(this.floatLayerOutput));
        result.setFloatLayerSums(arrayCopy(this.floatLayerSums));

        result.setLayerFeedCounts(EngineArray.arrayCopy(super.getLayerFeedCounts()));

        result.setContextTargetOffset(EngineArray.arrayCopy(super.getContextTargetOffset()));

        result.setContextTargetSize(EngineArray.arrayCopy(super.getContextTargetSize()));

        result.setLayerContextCount(EngineArray.arrayCopy(super.getLayerContextCount()));

        result.setBiasActivation(EngineArray.arrayCopy(super.getBiasActivation()));

        result.setOutputCount(super.getOutputCount());
        result.setWeightIndex(super.getWeightIndex());

        result.setFloatWeights(this.floatWeights);

        result.setActivationFunctions(new ActivationFunction[super.getActivationFunctions().length]);
        for(int i = 0; i < result.getActivationFunctions().length; i++) {
            result.getActivationFunctions()[i] = super.getActivationFunctions()[i].clone();
        }

        result.setBeginTraining(super.getBeginTraining());
        result.setEndTraining(super.getEndTraining());
    }

    /**
     * Calculate the error for this neural network. The error is calculated
     * using root-mean-square(RMS).
     * 
     * @param data
     *            The training set.
     * @return The error percentage.
     */
    public final double calculateError(final FloatMLDataSet data) {
        final ErrorCalculation errorCalculation = new ErrorCalculation();

        final double[] actual = new double[this.getOutputCount()];
        final FloatMLDataPair pair = BasicFloatMLDataPair.createPair(data.getInputSize(), data.getIdealSize());

        final double[] ideals = new double[pair.getIdealArray().length];

        for(int i = 0; i < data.getRecordCount(); i++) {
            data.getRecord(i, pair);
            compute(pair.getInputArray(), actual);
            for(int j = 0; j < ideals.length; j++) {
                ideals[j] = pair.getIdealArray()[j];
            }

            errorCalculation.updateError(actual, ideals, pair.getSignificance());
        }
        return errorCalculation.calculate();
    }

    /**
     * Copy a double array.
     * 
     * @param input
     *            The array to copy.
     * @return The result of the copy.
     */
    public static float[] arrayCopy(final float[] input) {
        final float[] result = new float[input.length];
        arrayCopy(input, result);
        return result;
    }

    /**
     * Completely copy one array into another.
     * 
     * @param src
     *            Source array.
     * @param dst
     *            Destination array.
     */
    public static void arrayCopy(final float[] src, final float[] dst) {
        System.arraycopy(src, 0, dst, 0, src.length);
    }

    /**
     * Copy an array of doubles.
     * 
     * @param source
     *            The source.
     * @param sourcePos
     *            The source index.
     * @param target
     *            The target.
     * @param targetPos
     *            The target index.
     * @param length
     *            The length.
     */
    public static void arrayCopy(final float[] source, final int sourcePos, final float[] target, final int targetPos,
            final int length) {
        System.arraycopy(source, sourcePos, target, targetPos, length);

    }

    /**
     * Decode the specified data into the weights of the neural network. This
     * method performs the opposite of encodeNetwork.
     * 
     * @param data
     *            The data to be decoded.
     */
    public void decodeNetwork(final float[] data) {
        if(data.length != this.floatWeights.length) {
            throw new EncogError("Incompatable weight sizes, can't assign length=" + data.length + " to length="
                    + floatWeights.length);
        }
        this.floatWeights = data;

    }

    /**
     * Set the weights.
     * 
     * @param weights
     *            The weights.
     */
    public void setWeights(final float[] weights) {
        this.floatWeights = weights;
    }

    /**
     * Set the layer sums.
     * 
     * @param layerSums
     *            The layer sums.
     */
    public void setLayerSums(float[] layerSums) {
        this.setFloatLayerSums(layerSums);
    }

    /**
     * @return the floatLayerSums
     */
    public float[] getFloatLayerSums() {
        return floatLayerSums;
    }

    /**
     * @param floatLayerSums
     *            the floatLayerSums to set
     */
    public void setFloatLayerSums(float[] floatLayerSums) {
        this.floatLayerSums = floatLayerSums;
    }

    /**
     * @return the floatLayerDropoutRates
     */
    public float[] getLayerDropoutRates() {
        return floatLayerDropoutRates;
    }

    /**
     * @param floatLayerDropoutRates
     *            the floatLayerDropoutRates to set
     */
    public void setLayerDropoutRates(float[] floatLayerDropoutRates) {
        this.floatLayerDropoutRates = floatLayerDropoutRates;
    }

    /**
     * @return the floatLayerOutput
     */
    public float[] getFloatLayerOutput() {
        return floatLayerOutput;
    }

    /**
     * @param floatLayerOutput
     *            the floatLayerOutput to set
     */
    public void setFloatLayerOutput(float[] floatLayerOutput) {
        this.floatLayerOutput = floatLayerOutput;
    }

    /**
     * @return the floatWeights
     */
    public float[] getFloatWeights() {
        return floatWeights;
    }

    /**
     * @param floatWeights
     *            the floatWeights to set
     */
    public void setFloatWeights(float[] floatWeights) {
        this.floatWeights = floatWeights;
    }

}
