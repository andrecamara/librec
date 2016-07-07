package librec.util;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class GaussianKernel {

	private RealMatrix kernel;
	private double[][] pureKernel;
	private double sigma = 1.0;
	double[][] matriz = null;

	public GaussianKernel(double[][] matriz) {
			this.matriz = matriz;
		}

	public GaussianKernel(double[][] matriz, double sigma) {
			this.matriz = matriz;
			this.sigma = sigma;
		}

	public void calculateKernel() {
		if (matriz != null) {
			int qtdLinhas = this.matriz.length;
			int qtdColunas = this.matriz[0].length;
			double[][] kernel = new double[qtdLinhas][qtdLinhas];
			double[] linha;

			int count;
			int count2 = 0;

			for (double[] ds : matriz) {
				count = 0;
				linha = new double[qtdLinhas];
				for (double[] d : matriz) {
					double sum = 0.0;
					for (int i = 0; i < qtdColunas; i++) {
						sum = sum + Math.pow((ds[i] - d[i]), 2);
					}
					// linha[count] =
					// Math.exp(-(Math.sqrt(sum)/(2*this.sigma)));
					double valor = Math.exp(-(sum / (2 * Math.pow(this.sigma, 2))));
					linha[count] = valor;
					count++;
				}
				kernel[count2] = linha;
				count2++;
			}
			this.pureKernel = kernel;
			this.kernel = new Array2DRowRealMatrix(kernel, true);
		}
	}

	public RealMatrix getKernel() {
		return kernel;
	}

	public void setKernel(RealMatrix kernel) {
		this.kernel = kernel;
	}

	public double getSigma() {
		return sigma;
	}

	public void setSigma(double sigma) {
		this.sigma = sigma;
	}

	public double[][] getMatriz() {
		return matriz;
	}

	public void setMatriz(double[][] matriz) {
		this.matriz = matriz;
	}

}
