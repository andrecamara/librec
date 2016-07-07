package librec.ranking;

import java.io.BufferedReader;
import java.io.FileReader;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Iterator;
import java.util.Set;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import librec.data.Configuration;
import librec.data.SparseMatrix;
import librec.intf.IterativeRecommender;
import librec.util.GaussianKernel;
import librec.util.KroneckerOperation;
import librec.util.LineConfiger;
import librec.util.Strings;

/**
 * Laarhoven / Pahikkala, <strong>Kronecker Regularized Least Squares</strong>
 * 
 */
@Configuration("lambda")
public class KronRLS extends IterativeRecommender {

	private RealMatrix kernelUsuarios;
	private RealMatrix kernelItems;
	private RealMatrix A;
	private RealMatrix myTrainMatrix;
	private double lambda;
//	private RealMatrix myTestMatrix;

	public KronRLS(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		isRankingPred = true;
		// checkBinary();

	}

	@Override
	protected void initModel() throws Exception {
		super.initModel();

		// remove users / items not in trainMatrix
		// int[] col_idx = trainMatrix.getColumnIndices();
		// int[] row_idx = trainMatrix.getRowPointers();
		//
		// Integer[] n_col_idx = ArrayUtils.toObject(col_idx);
		// Integer[] n_row_idx = ArrayUtils.toObject(row_idx);
		//
		// HashSet<Integer> uniqColIdx = new HashSet<Integer>();
		// uniqColIdx.addAll(Arrays.asList(n_col_idx));
		//
		// HashSet<Integer> uniqRowIdx = new HashSet<Integer>();
		// uniqRowIdx.addAll(Arrays.asList(n_row_idx));

		if (algoOptions.contains("-user_data")) {
			String usersData = algoOptions.getString("-user_data");
			System.out.println("Calculating user kernel...");
			// TODO vectorize GaussianKernel calculation
			GaussianKernel kUsers = new GaussianKernel(this.readUsersData(usersData));
			kUsers.calculateKernel();
			this.kernelUsuarios = kUsers.getKernel();
		}
		if (algoOptions.contains("-item_data")) {
			String itemsData = algoOptions.getString("-item_data");
			System.out.println("Calculating items kernel...");
			// double[][] temp = this.readItemsData(itemsData);
			// TODO vectorize GaussianKernel calculation
			GaussianKernel kItems = new GaussianKernel(this.readItemsData(itemsData));
			kItems.calculateKernel();
			this.kernelItems = kItems.getKernel();
		}
		if (algoOptions.contains("-lambda")) {
			lambda = algoOptions.getDouble("-lambda");
		}
		
		// load train matrix and test matrix internally (libRec BUG?)
		String trainFile = cf.getString("dataset.ratings.lins");
//			String trainFile = algoOptions.getString("-ratings") + "u1.base";
//			String testFile  = algoOptions.getString("-ratings") + "u1.test";
		myTrainMatrix = new Array2DRowRealMatrix(readRatingData(trainFile));
//			myTestMatrix  = new Array2DRowRealMatrix(readRatingData(testFile));
	}

	@Override
	protected void buildModel() throws Exception {
		this.A = null;
		
		if (kernelUsuarios != null && kernelItems != null && myTrainMatrix != null) {
			
			System.out.println("Calculating eigendecompositions...");
			EigenDecomposition eigenDecompositionUser = new EigenDecomposition(this.kernelUsuarios);
			EigenDecomposition eigenDecompositionItems = new EigenDecomposition(this.kernelItems);

			RealMatrix lU = eigenDecompositionUser.getD();
			RealMatrix lI = eigenDecompositionItems.getD();

			RealMatrix qU = eigenDecompositionUser.getV();
			RealMatrix qI = eigenDecompositionItems.getV();

			int count = 0;
			double[] diagonalLU = new double[lU.getData().length];
			for (double[] d : lU.getData()) {
				double value = d[count];
				diagonalLU[count] = value;
				count++;
			}

			double[][] matrixDiagonalLU = new Array2DRowRealMatrix(diagonalLU).getData();

			count = 0;
			double[] diagonalLI = new double[lI.getData().length];
			for (double[] d : lI.getData()) {
				double value = d[count];
				diagonalLI[count] = value;
				count++;
			}

			System.out.println("Diagonais done --------------");

			double[][] matrixDiagonalLIT = new Array2DRowRealMatrix(diagonalLI).transpose().getData();

			// TODO use DenseMatrix.kroneckerProduct instead
			System.out.println("Start kronecker product of diagonal --------------");
			double[][] l = KroneckerOperation.product(matrixDiagonalLIT, matrixDiagonalLU);

			RealMatrix matrixL = new Array2DRowRealMatrix(l);

			System.out.println("Start calculating inverse --------------");
			RealMatrix matrixInverse = new Array2DRowRealMatrix(
					matrixL.getRowDimension(), 
					matrixL.getColumnDimension());
			for (int i = 0; i < matrixL.getRowDimension(); i++) {
				RealVector rv1 = new ArrayRealVector(matrixL.getRow(i));
				RealVector rv2 = new ArrayRealVector(matrixL.getRow(i)).mapAddToSelf(lambda);
				rv1.ebeDivide(rv2);
				matrixInverse.setRowVector(i, rv1);
			}
			
			System.out.println("end of inverse calculation");

//			RealMatrix tmp = RealMatrixUtils.sparseMatrix2RealMatrix(trainMatrix);
			RealMatrix m1 = qU.transpose().multiply(myTrainMatrix).multiply(qI);

//			double[][] hadamard = new double[m1.getRowDimension()][m1.getColumnDimension()];
			System.out.println("Start hadamard product  --------------");
			RealMatrix m2 = new Array2DRowRealMatrix(m1.getRowDimension(), m1.getColumnDimension());
			for (int i = 0; i < m1.getRowDimension(); i++) {
				RealVector rv1 = new ArrayRealVector(m1.getRow(i));
				RealVector rv2 = new ArrayRealVector(matrixInverse.getRow(i));
				rv1.ebeMultiply(rv2);
				m2.setRowVector(i, rv1);
			}
			System.out.println("Start calculating result  --------------");
			A = qU.multiply(m2).multiply(qI.transpose());

		} else {
			throw new Exception("Os dados precisam ser carregados");
			// System.out.println("Os dados precisam ser carregados");
		}
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { lambda});
	}

	/*
	 * KRONRLS-specific methods
	 */
	public double[][] readUsersData(String diretorioUser) {
		ArrayList<String> listaUsers = this.readFileData(diretorioUser);
		ArrayList<ArrayList<Integer>> listaUsersI = new ArrayList<ArrayList<Integer>>();

		// listaUsers = filterList(listaUsers, row_idx);

		for (String s : listaUsers) {
			ArrayList<Integer> user = new ArrayList<Integer>();
			String i = s.replace("|", ":");

			i = i.replace("F", "0");
			i = i.replace("M", "1");

			String[] split = i.split(":");

			boolean linhaValida = true;

			for (int j = 0; j < split.length; j++) {
				if (j == 3) {
					String transformarDados = "";
					if (split[j].equals("administrator")) {
						transformarDados = "1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0";
					} else if (split[j].equals("artist")) {
						transformarDados = "0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0";
					} else if (split[j].equals("doctor")) {
						transformarDados = "0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0";
					} else if (split[j].equals("educator")) {
						transformarDados = "0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0";
					} else if (split[j].equals("engineer")) {
						transformarDados = "0|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0";
					} else if (split[j].equals("entertainment")) {
						transformarDados = "0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0";
					} else if (split[j].equals("executive")) {
						transformarDados = "0|0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0";
					} else if (split[j].equals("healthcare")) {
						transformarDados = "0|0|0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0";
					} else if (split[j].equals("homemaker")) {
						transformarDados = "0|0|0|0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0";
					} else if (split[j].equals("lawyer")) {
						transformarDados = "0|0|0|0|0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0";
					} else if (split[j].equals("librarian")) {
						transformarDados = "0|0|0|0|0|0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0";
					} else if (split[j].equals("marketing")) {
						transformarDados = "0|0|0|0|0|0|0|0|0|0|0|1|0|0|0|0|0|0|0|0|0";
					} else if (split[j].equals("none")) {
						transformarDados = "0|0|0|0|0|0|0|0|0|0|0|0|1|0|0|0|0|0|0|0|0";
					} else if (split[j].equals("other")) {
						transformarDados = "0|0|0|0|0|0|0|0|0|0|0|0|0|1|0|0|0|0|0|0|0";
					} else if (split[j].equals("programmer")) {
						transformarDados = "0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|0|0|0|0|0|0";
					} else if (split[j].equals("retired")) {
						transformarDados = "0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|0|0|0|0|0";
					} else if (split[j].equals("salesman")) {
						transformarDados = "0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|0|0|0|0";
					} else if (split[j].equals("scientist")) {
						transformarDados = "0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|0|0|0";
					} else if (split[j].equals("student")) {
						transformarDados = "0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|0|0";
					} else if (split[j].equals("technician")) {
						transformarDados = "0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|0";
					} else if (split[j].equals("writer")) {
						transformarDados = "0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1";
					}

					String novoI = transformarDados.replace("|", ":");

					String[] novoSplit = novoI.split(":");

					for (String string : novoSplit) {
						int valor;

						try {
							valor = Integer.parseInt(string);
						} catch (Exception e) {
							e.printStackTrace();
							valor = 0;
							linhaValida = false;

						}
						user.add(valor);
					}
				} else if (j != 0 && j != 4) {
					int valor;

					try {
						valor = Integer.parseInt(split[j]);
					} catch (Exception e) {
						e.printStackTrace();
						valor = 0;
						linhaValida = false;

					}
					user.add(valor);
				}
			}

			if (linhaValida) {
				listaUsersI.add(user);
			}
		}

		int ulinhas = listaUsersI.size();
		int ucolunas = listaUsersI.get(0).size();
		double[][] usersMatriz = new double[ulinhas][ucolunas];

		for (int i = 0; i < ulinhas; i++) {
			for (int j = 0; j < ucolunas; j++) {
				usersMatriz[i][j] = Double.parseDouble(listaUsersI.get(i).get(j).toString());
			}
		}

		return usersMatriz;
	}

	@SuppressWarnings("rawtypes")
	private ArrayList<String> filterList(ArrayList<String> lista, Set row_idx) {
		ArrayList res = new ArrayList();
		for (Iterator iterator = row_idx.iterator(); iterator.hasNext();) {
			Integer row = (Integer) iterator.next();
			res.add(lista.get(row));
		}
		return res;
	}

	@SuppressWarnings("deprecation")
	public double[][] readItemsData(String diretorioItem) {
		SimpleDateFormat sdf = new SimpleDateFormat("ddMMyyyy");
		ArrayList<String> listaItems = this.readFileData(diretorioItem);
		ArrayList<ArrayList<Integer>> listaItemsI = new ArrayList<ArrayList<Integer>>();

		// listaItems = filterList(listaItems, row_idx);

		for (String s : listaItems) {
			ArrayList<Integer> item = new ArrayList<Integer>();
			String i = s.replace("|", "	");

			String[] split = i.split("	");

			boolean linhaValida = true;

			for (int j = 0; j < split.length; j++) {
				int valor;
				if (j == 2) {
					try {
						String d = split[j];
						String[] date = d.split("-");
						Date dataLancamento = new Date(date[1] + "/" + date[0] + "/" + date[2]);

						valor = (int) dataLancamento.getYear();

						item.add(valor);
					} catch (Exception e) {
						System.out.println(s);
						linhaValida = false;
					}
				}
				if (j != 0 && j != 3 && j != 1 && j != 2 && j != 4) {
					valor = Integer.parseInt(split[j]);
					item.add(valor);
				}
			}

			if (linhaValida) {
				listaItemsI.add(item);
			}
		}

		int ilinhas = listaItemsI.size();
		int icolunas = listaItemsI.get(0).size();
		double[][] itemsMatriz = new double[ilinhas][icolunas];

		for (int i = 0; i < ilinhas; i++) {
			for (int j = 0; j < icolunas; j++) {
				itemsMatriz[i][j] = Double.parseDouble(listaItemsI.get(i).get(j).toString());
			}
		}

		return itemsMatriz;

	}

	public ArrayList<String> readFileData(String diretorioData) {
		ArrayList<String> lista = new ArrayList<String>();
		FileReader data;
		String linha;

		try {
			data = new FileReader(diretorioData);
			BufferedReader lerArquivo = new BufferedReader(data);

			linha = lerArquivo.readLine();
			lista.add(linha);
			while (linha != null) {
				linha = lerArquivo.readLine();
				if (linha != null) {
					lista.add(linha);
				}
			}

			return lista;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	public double[][] readRatingData(String diretorioData) {
		ArrayList<ArrayList<Integer>> listaDados = new ArrayList<ArrayList<Integer>>();
		ArrayList<String> listaData = this.readFileData(diretorioData);
		for (String s : listaData) {
			ArrayList<Integer> dados = new ArrayList<Integer>();
			String[] split = s.split("\t");
//			System.out.println(s);
			boolean linhaValida = true;

			for (int j = 0; j < split.length; j++) {
				if (j != 3) {
					int valor;

					try {
						valor = Integer.parseInt(split[j]);
					} catch (Exception e) {
						e.printStackTrace();
						valor = 0;
						linhaValida = false;

					}
					dados.add(valor);
				}
			}

			if (linhaValida) {
//				System.out.println(dados);
				if (dados.get(0) <= this.kernelUsuarios.getRowDimension()
						&& dados.get(1) <= this.kernelItems.getRowDimension()) {
					listaDados.add(dados);
				}
			}

		}

		double[][] dadosMatriz = new double[this.kernelUsuarios.getRowDimension()][this.kernelItems.getRowDimension()];

		for (ArrayList<Integer> d : listaDados) {
			dadosMatriz[d.get(0) - 1][d.get(1) - 1] = Double.parseDouble(d.get(2).toString());
		}

		return dadosMatriz;
	}
//	
//	@Override
//	protected double predict(int u, int j) {
//		return this.A.getEntry(u, j);
//	}
}
