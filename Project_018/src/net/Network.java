package net;

import java.util.Random;

public class Network {
	
	Random r = new Random(0);
	
	private double modifi = .01;
	private double l_rate = .01;
	private double moment = .01;
	private double leak_r = .01;
	
	private int[] widths;
	private int depth;
	
	private double[][][] weights;
	private double[][][] weightsdiff;
	private double[][] bias;
	private double[][] biasdiff;

	public Network(int[] widths) {
		this.widths = widths;
		depth = widths.length;
		weights = new double[depth][][];
		weightsdiff = new double[depth][][];
		bias = new double[depth][];
		biasdiff = new double[depth][];
		
		for(int d = 0; d < depth; d++) {
			weights[d] = new double[getWidth(d)][];
			weightsdiff[d] = new double[getWidth(d)][];
			bias[d] = new double[getWidth(d)];
			biasdiff[d] = new double[getWidth(d)];	
			for(int w = 0; w < getWidth(d); w++) {
				weights[d][w] = new double[getWidth(d - 1)];
				weightsdiff[d][w] = new double[getWidth(d - 1)];
				for(int w2 = 0; w2 < getWidth(d-1); w2++) {
					weights[d][w][w2] = randomModified();
				}
			}
		}
	}
	
	private void test() {
		long start = System.nanoTime();
		
		int epochs = 1000000;
		int inputs = widths[0];
		int outputs = widths[depth - 1];
		
		double[] in = new double[inputs];
		double[] out = new double[outputs];
		
		for(int i = 0; i < epochs; i++) {
			for(int j = 0; j < inputs; j++) {
				in[j] = r.nextDouble();
			}
			for(int j = 0; j < outputs; j++) {
				out[j] = func(in[j]);
			}
			train(in, out);
		}
		for(int j = 0; j < inputs; j++) {
			in[j] = r.nextDouble();
		}
		out = forwardPass(in);
		for(int i = 0; i < out.length; i++) {
			System.out.println(out[i]+" Expected: "+func(in[i]));
		}
		
		System.out.println("Time: "+ (System.nanoTime() - start) / 1e9f);
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
				double Sum = (d == depth - 1)? (y[w] - h[d][w]) : 0;
				for (int w2 = 0; w2 < getWidth(d+1); w2++) {
					Sum += weights[d+1][w2][w] * SignalError[d+1][w2];
				}
				SignalError[d][w] = Sum * ActivationFunction(h[d][w], true);
			}
		}
		for (int d = depth - 1; d > 0; d--) {
			for (int w = 0; w < getWidth(d); w++) {
				biasdiff[d][w] = l_rate * SignalError[d][w] + moment * biasdiff[d][w];
				bias[d][w] += biasdiff[d][w];
				for (int w2 = 0; w2 < getWidth(d-1); w2++) {
					weightsdiff[d][w][w2] = l_rate * SignalError[d][w] * h[d-1][w2] + moment * weightsdiff[d][w][w2];
					weights[d][w][w2] += weightsdiff[d][w][w2];
				}
			}
		}
	}
	
	public int getWidth(int d) {
		return (d < 0 || d > depth - 1)? 0 : widths[d];
	}

	private double randomModified() {
		return r.nextDouble() * (r.nextBoolean()? modifi : -modifi);
	}
	
	private double ActivationFunction(double x, boolean derivative) {
		return ReLU(x, derivative);
	}
	
	private double ReLU(double x, boolean derivative) {
		if(derivative)
			return (x > 0)? 1 : leak_r;
		else
			return (x > 0)? x : 0;
	}
	
	public double func(double in) {
		return Math.sin(in);
	}
	
	public static void main(String[] args) {
		Network n = new Network(new int[]{1, 2, 1});
		n.test();
	}
}
