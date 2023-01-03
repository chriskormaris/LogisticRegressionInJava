package data;

public class Constants {

	// number of words in feature dictionary 
	public static final int NUM_FEATURES = 1000;

	public final static String SPAM_TRAIN_PATH = ResourceLoader.load("spam-train").getPath().substring(1);
	public final static String HAM_TRAIN_PATH = ResourceLoader.load("nonspam-train").getPath().substring(1);
	public final static String SPAM_TEST_PATH = ResourceLoader.load("spam-test").getPath().substring(1);
	public final static String HAM_TEST_PATH = ResourceLoader.load("nonspam-train").getPath().substring(1);
	public final static String FEATURE_DICTIONARY_PATH = ResourceLoader.load("feature_dictionary.txt").getPath().substring(1);

	public final static int LABEL_SPAM = 1;
	public final static int LABEL_HAM = 0;

	public final static String LABEL_SPAM_STRING = "SPAM";
	public final static String LABEL_HAM_STRING = "HAM";

}
