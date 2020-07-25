package data;

public class Constants {
	
	// number of words in feature dictionary 
	public static final int NO_OF_FEATURES = 1000;
	
	public final static String SPAM_TRAIN_PATH	= "LingspamDataset/spam-train";
	public final static String HAM_TRAIN_PATH	= "LingspamDataset/nonspam-train";
	public final static String SPAM_TEST_PATH	= "LingspamDataset/spam-test";
	public final static String HAM_TEST_PATH	= "LingspamDataset/nonspam-test";
	public final static String FEATURE_DICTIONARY_PATH	= "feature_dictionary.txt";

	public final static int LABEL_SPAM = 1;
	public final static int LABEL_HAM  = 0;
	
	public final static String LABEL_SPAM_STRING	= "SPAM";
	public final static String LABEL_HAM_STRING  	= "HAM";

}
