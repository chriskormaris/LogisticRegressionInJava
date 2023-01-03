package data;

public class Constants {

	public static final int NUM_PIXELS = 784;

	public final static String TRAIN1_PATH = ResourceLoader.load("./train1.txt").getPath().substring(1);
	public final static String TRAIN7_PATH = ResourceLoader.load("./train7.txt").getPath().substring(1);
	public final static String TEST1_PATH = ResourceLoader.load("./test1.txt").getPath().substring(1);
	public final static String TEST7_PATH = ResourceLoader.load("./test7.txt").getPath().substring(1);

	public final static int LABEL_ONE = 1;
	public final static int LABEL_SEVEN = 0;

	public final static String LABEL_ONE_STRING = "1";
	public final static String LABEL_SEVEN_STRING = "7";

}
