package librec.main;

import librec.util.FileIO;
import librec.util.Logs;

public class EvaluateKronRLS {
	public static void main(String[] args) {
		String dirPath = FileIO.makeDirPath("demo");
		Logs.config(dirPath + "log4j.xml", true);

		// set the folder path for configuration files
		String configDirPath = FileIO.makeDirPath(dirPath, "config");
		String configFile = "KronRLS.conf";
		
		// run algorithm
		LibRec librec = new LibRec();
		librec.setConfigFiles(configDirPath + configFile);
		
		try {
			librec.execute(args);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
