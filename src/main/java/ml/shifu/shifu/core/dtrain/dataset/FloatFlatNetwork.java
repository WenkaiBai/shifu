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

import ml.shifu.shifu.core.dtrain.nn.BasicDropoutLayer;

import java.util.Arrays;
import java.util.HashSet;

import org.encog.neural.flat.FlatLayer;
import org.encog.neural.flat.FlatNetwork;
import org.encog.util.EngineArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * To solve float input and output types.
 */
public class FloatFlatNetwork extends FlatNetwork implements Cloneable {

    protected static final Logger LOG = LoggerFactory.getLogger(FloatFlatNetwork.class);
	
    private static final long serialVersionUID = -7208969306860840672L;

    /**
     * The dropout rate for each layer.
     */
    private double[] layerDropoutRates;

    /**
     * Indices of dropout nodes
     */
    private HashSet<Integer> dropoutNodes;

    /**
     * after dropout mask, the layer output
     */
    private double[] maskedLayerOutput;
    
	public FloatFlatNetwork() {
        this.layerDropoutRates = new double[0];
    }

    public FloatFlatNetwork(final FlatLayer[] layers, final HashSet<Integer> dropoutNodes) {
        this(layers, dropoutNodes, true);
    }

    public FloatFlatNetwork(final FlatLayer[] layers, final HashSet<Integer> dropoutNodes, boolean dropout) {
        init(layers, dropoutNodes, dropout);
    }

    private void init(FlatLayer[] layers, final HashSet<Integer> dropoutNodes, boolean dropout) {
        super.init(layers);
        this.maskedLayerOutput = EngineArray.arrayCopy(getLayerOutput());
        
        final int layerCount = layers.length;
        if(dropout) {
            this.setLayerDropoutRates(new double[layerCount]);

            int index = 0;
            for(int i = layers.length - 1; i >= 0; i--) {
                final FlatLayer layer = layers[i];
                if(layer instanceof BasicDropoutLayer) {
                	this.layerDropoutRates[index] = ((BasicDropoutLayer) layer).getDropout();
                }
                index += 1;
            }
            
            this.dropoutNodes = dropoutNodes;
        } else {
            this.setLayerDropoutRates(new double[0]);
            this.dropoutNodes = null;
        }

    }

    public void compute(float[] input, double[] output) {
        final int sourceIndex = getLayerOutput().length - getLayerCounts()[getLayerCounts().length - 1];

        for(int i = 0; i < getInputCount(); i++) {
            getLayerOutput()[i + sourceIndex] = input[i];
        }

        for(int i = this.getLayerIndex().length - 1; i > 0; i--) {
            computeLayer(i);
        }

        // update context values
        final int offset = getContextTargetOffset()[0];

        for(int x = 0; x < getContextTargetSize()[0]; x++) {
            this.getLayerOutput()[offset + x] = this.getLayerOutput()[x];
        }

        System.arraycopy(getLayerOutput(), 0, output, 0, this.getOutputCount());
    }
    
    public void compute(float[] input, float[] output) {
        final int sourceIndex = getLayerOutput().length - getLayerCounts()[getLayerCounts().length - 1];

        for(int i = 0; i < getInputCount(); i++) {
            getLayerOutput()[i + sourceIndex] = input[i];
        }

        for(int i = getLayerIndex().length - 1; i > 0; i--) {
            computeLayer(i);
        }

        // update context values
        final int offset = getContextTargetOffset()[0];

        for(int x = 0; x < getContextTargetSize()[0]; x++) {
            this.getLayerOutput()[offset + x] = this.getLayerOutput()[x];
        }

        // copy to float output array
        for(int i = 0; i < getOutputCount(); i++) {
            output[i] = (float) getLayerOutput()[i];
        }
    }

    public void computeWithDropout(float[] input, double[] dropoutNetworkOutput) {
        final int sourceIndex = getLayerOutput().length - getLayerCounts()[getLayerCounts().length - 1];
        final int inputLayerIndex = this.getLayerIndex().length - 1;
 
        LOG.info("computeWithDropout: sourceIndex={}; inputLayerIndex={}", sourceIndex, inputLayerIndex); 
        if (dropoutNodes != null) {
        	LOG.info("                    dropoutNodes={}", dropoutNodes.toString());
        }
        
        for(int i = 0; i < getInputCount(); i++) {
        	getLayerOutput()[i + sourceIndex] = input[i];
        	
        	if(isDropoutEnabled(inputLayerIndex)) {
            	if (dropoutNodes.contains((i + sourceIndex))) {
            		this.maskedLayerOutput[i + sourceIndex] = 0;
            	} else {
            		this.maskedLayerOutput[i + sourceIndex] = input[i] / (1d - getLayerDropoutRate(inputLayerIndex));
            	}
        	} else {
                this.maskedLayerOutput[i + sourceIndex] = input[i];
        	}
        }

        for(int i = inputLayerIndex; i > 0; i--) {
        	computeLayerWithDropout(i);
        }

        // update context values
        final int offset = getContextTargetOffset()[0];

        for(int x = 0; x < getContextTargetSize()[0]; x++) {
        	this.maskedLayerOutput[offset + x] = this.maskedLayerOutput[x];
            this.getLayerOutput()[offset + x] = this.getLayerOutput()[x];
        }

        LOG.info("                    getLayerOutput={}", Arrays.toString(getLayerOutput()));
        LOG.info("                    maskedLayerOutput={}", Arrays.toString(this.maskedLayerOutput));
        
        System.arraycopy(getLayerOutput(), 0, dropoutNetworkOutput, 0, this.getOutputCount());
    }
    
    public double getLayerDropoutRate(final int currentLayer) {
        if(this.layerDropoutRates.length > currentLayer) {
            return this.layerDropoutRates[currentLayer];
        } else {
            return 0d;
        }
    }

    protected void computeLayerWithDropout(final int currentLayer) {
    	final double dropoutRate = getLayerDropoutRate(currentLayer - 1);
    	
    	final int inputIndex = super.getLayerIndex()[currentLayer];
        final int outputIndex = super.getLayerIndex()[currentLayer - 1];
        final int inputSize = super.getLayerCounts()[currentLayer];
        final int outputSize = super.getLayerFeedCounts()[currentLayer - 1];
        
        LOG.info("                    inputIndex={},  outputIndex={},  inputSize={},  outputSize={}", 
        		inputIndex, outputIndex, inputSize, outputSize);   
        
        int index = super.getWeightIndex()[currentLayer - 1];

        final int limitX = outputIndex + outputSize;
        final int limitY = inputIndex + inputSize;

        // weight values
        for(int x = outputIndex; x < limitX; x++) {
            double sum = 0;
            for(int y = inputIndex; y < limitY; y++) {
                sum += super.getWeights()[index++] * this.maskedLayerOutput[y];
            }
            super.getLayerSums()[x] = sum;
            super.getLayerOutput()[x] = sum;
        }

        super.getActivationFunctions()[currentLayer - 1].activationFunction(super.getLayerOutput(), outputIndex,
                outputSize);
        
        // choose dropout nodes and make its output to be zero
        if(isDropoutEnabled(currentLayer - 1)) {
            // weight values
            for(int x = outputIndex; x < limitX; x++) {
                if (this.dropoutNodes.contains(x)) {
                	this.maskedLayerOutput[x] = 0;
                } else {
                	this.maskedLayerOutput[x] = (super.getLayerOutput()[x] /  (1d - dropoutRate));
                }
            }
        } else {
        	System.arraycopy(maskedLayerOutput, outputIndex, super.getLayerOutput(), outputIndex, limitX);
        }

        // update context values
        final int offset = super.getContextTargetOffset()[currentLayer];

        for(int x = 0; x < super.getContextTargetSize()[currentLayer]; x++) {
            super.getLayerOutput()[offset + x] = super.getLayerOutput()[outputIndex + x];
            this.maskedLayerOutput[offset + x] = this.maskedLayerOutput[outputIndex + x];
        }
    }
    
    /**
     * Clone the network.
     * 
     * @return A clone of the network.
     */
    @Override
    public FloatFlatNetwork clone() {
        final FloatFlatNetwork result = new FloatFlatNetwork();
        super.cloneFlatNetwork(result);
        
        if (this.layerDropoutRates != null) {
        	result.setLayerDropoutRates( EngineArray.arrayCopy(this.layerDropoutRates));	
        }
        if (this.dropoutNodes != null) {
        	result.setDropoutNodes(new HashSet<Integer>(this.dropoutNodes));
        }
        if (this.maskedLayerOutput != null) {
        	result.setMaskedLayerOutput(EngineArray.arrayCopy(this.maskedLayerOutput));
        }
        
        return result;
    }

	/**
	 * is dropout enabled on certain layer
	 *  if dropoutNodes == null, it means there are some accident happened, we should not apply dropout logic
	 * @param layerIndex layer index
	 * @return if this layer need to be dropout or not
	 */
    public boolean isDropoutEnabled(int layerIndex) {
    	return (Double.compare(getLayerDropoutRate(layerIndex), 0d) != 0 && dropoutNodes != null);
    }
    
	/**
	 *  is dropout enabled on entire network
	 * @return if this layer need to be dropout or not
	 */
    public boolean isDropoutEnabled() {
    	return (dropoutNodes != null && !dropoutNodes.isEmpty());
    }
    
    /**
     * @param layerDropoutRates
     *            the layerDropoutRates to set
     */
    public void setLayerDropoutRates(double[] layerDropoutRates) {
        this.layerDropoutRates = layerDropoutRates;
    }
    
    /**
     * @return dropout node indices
     */
    public HashSet<Integer> getDropoutNodes() {
		return dropoutNodes;
	}

	public void setDropoutNodes(HashSet<Integer> dropoutNodes) {
		this.dropoutNodes = dropoutNodes;
	}

	/**
	 * @return after dropout mask, output of each neuron
	 */
	public double[] getMaskedLayerOutput() {
		return maskedLayerOutput;
	}

	public void setMaskedLayerOutput(double[] maskedLayerOutput) {
		this.maskedLayerOutput = maskedLayerOutput;
	}
}
