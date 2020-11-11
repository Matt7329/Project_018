package net;

import java.util.Random;

public class Network {
	
	Random r = new Random();
	
	private double mod = 0.1f;
	private double learning_rate = 0.01f;

	private int[] widths;
	private int depth;
	
	private double[][][] weights;
	private double[][] bias;

	public Network(int[] widths) {
		this.widths = widths;
		depth = widths.length;
		weights = new double[depth][][];
		bias = new double[depth][];
		for(int d = 0; d < depth; d++) {
			bias[d] = new double[getWidth(d)];
			weights[d] = new double[getWidth(d)][];
			for(int w = 0; w < getWidth(d); w++) {
				weights[d][w] = new double[getWidth(d - 1)];
				for(int w2 = 0; w2 < getWidth(d-1); w2++) {
					weights[d][w][w2] = randomModified();
				}
			}
		}
	}
	
	public double[] forwardPass(double[] x) {
		double[][] h = new double[depth][];
		h[0] = x;
		for(int d = 1; d < depth; d++) {		
			h[d] = new double[getWidth(d)];
			for(int w = 0; w < getWidth(d); w++) {
				for(int w2 = 0; w2 < getWidth(d-1); w2++) {
					h[d][w] += h[d-1][w2] * weights[d][w][w2];
				}
				h[d][w] = ActivationFunction(h[d][w] + bias[d][w], false);
			}
		}
		return h[depth - 1];
	}
	
	public void train(double[] x, double[] y) {
		double[][] h = new double[depth][];
		h[0] = x;
		for(int d = 1; d < depth; d++) {		
			h[d] = new double[getWidth(d)];
			for(int w = 0; w < getWidth(d); w++) {
				for(int w2 = 0; w2 < getWidth(d-1); w2++) {
					h[d][w] += h[d-1][w2] * weights[d][w][w2];
				}
				h[d][w] = ActivationFunction(h[d][w] + bias[d][w], false);
			}
		}	
		double[][] SignalError = new double[depth][];		
		for (int d = depth - 1; d > 0; d--) {
			SignalError[d] = new double[getWidth(d)];
			for (int w = 0; w < getWidth(d); w++) {
				double Sum = (y[w] - h[d][w]);
				if(d < depth - 1) {
					Sum = 0;
					for (int w2 = 0; w2 < getWidth(d+1); w2++) {
						Sum += weights[d+1][w2][w] * SignalError[d+1][w2];
					}
				}
				SignalError[d][w] = Sum * ActivationFunction(h[d][w], true);
			}
		}
		for (int d = depth - 1; d > 0; d--) {
			for (int w = 0; w < getWidth(d); w++) {
				bias[d][w] += learning_rate * SignalError[d][w];
				for (int w2 = 0; w2 < getWidth(d-1); w2++) {
					weights[d][w][w2] += learning_rate * SignalError[d][w] * h[d-1][w2];
				}
			}
		}
	}
	
	public int getWidth(int d) {
		return (d < 0 || d > depth - 1)? 0 : this.widths[d];
	}

	private double randomModified() {
		return r.nextDouble() * (r.nextBoolean()? mod : -mod);
	}
	
	private double ActivationFunction(double x, boolean derivative) {
		return ReLU(x, derivative);
	}
	private double ReLU(double x, boolean derivative) {
		if(derivative)
			return (x > 0)? 1 : 0.01;
		else
			return (x > 0)? x : 0;
	}
	
	public static void main(String[] args) {
		Network n = new Network(new int[]{5, 5, 5});
		double[] in = new double[]{.5f, .5f, .5f, .5f, .5f};
		double[] out = new double[]{.3f, .4f, .5f, .6f, .7f};
		for(int i = 0; i < 10000000; i++) {
			n.train(in, out);
		}
		out = n.forwardPass(in);
		for(int i = 0; i < out.length; i++) {
			System.out.print(out[i]+" ");
		}
	}
}
