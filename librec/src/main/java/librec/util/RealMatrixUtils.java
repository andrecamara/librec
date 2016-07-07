package librec.util;

import java.util.Iterator;
import java.util.List;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import librec.data.SparseMatrix;

public class RealMatrixUtils {

	/**
	 * @return Kronecker product of two arbitrary matrices
	 */
	public static RealMatrix kroneckerProduct(RealMatrix M, RealMatrix N) {
		RealMatrix res = new Array2DRowRealMatrix
				(M.getRowDimension() * N.getRowDimension(), 
				M.getColumnDimension() * N.getColumnDimension());
		for (int i = 0; i < M.getRowDimension(); i++) {
			for (int j = 0; j < M.getColumnDimension(); j++) {
				double Mij = M.getEntry(i, j);
				// Mij*N
				for (int ni = 0; ni < N.getRowDimension(); ni++) {
					for (int nj = 0; nj < N.getColumnDimension(); nj++) {
						int row = i * N.getRowDimension() + ni;
						int col = j * N.getColumnDimension() + nj;

						res.setEntry(row, col, Mij * N.getEntry(ni, nj));
					}
				}
			}
		}

		return res;
	}

	/**
	 * @return Hadamard product of two matrices
	 */
	public static RealMatrix hadamardProduct(RealMatrix M, RealMatrix N) throws Exception {
		if (M.getRowDimension() != N.getRowDimension() || M.getColumnDimension() != N.getColumnDimension())
			throw new Exception("The dimensions of two matrices are not consistent!");

		RealMatrix res = new Array2DRowRealMatrix(M.getRowDimension(), M.getColumnDimension());

		for (int i = 0; i < M.getRowDimension(); i++) {
			for (int j = 0; j < M.getColumnDimension(); j++) {
				res.setEntry(i, j, M.getEntry(i, j) * N.getEntry(i, j));
			}
		}

		return res;
	}
	
	/**
	 * @return 
	 */
	public static RealMatrix sparseMatrix2RealMatrix(SparseMatrix m){
		RealMatrix res = new Array2DRowRealMatrix(m.numRows(), m.numColumns());
		
		for (int i = 0; i < m.numRows(); i++) {
			List cols = m.getColumns(i);
			for (Iterator iterator = cols.iterator(); iterator.hasNext();) {
				Integer col = (Integer) iterator.next();
				int j = col.intValue();
				res.setEntry(i, j, m.get(i,j));
			}
		}
		return res;
	}
	
//	/**
//	 * @return the inverse: m ./ (m + lambda) 
//	 */
//	public static RealMatrix inverse(RealMatrix m, double lambda){
//		RealMatrix res = new Array2DRowRealMatrix(m.numRows(), m.numColumns());
//		
//		for (int i = 0; i < m.numRows(); i++) {
//			List cols = m.getColumns(i);
//			for (Iterator iterator = cols.iterator(); iterator.hasNext();) {
//				Integer col = (Integer) iterator.next();
//				int j = col.intValue();
//				res.setEntry(i, j, m.get(i,j));
//			}
//		}
//		return res;
//	}
	
}
