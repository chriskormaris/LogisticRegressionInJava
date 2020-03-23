import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.Arrays;
import java.util.HashSet;


public class Utilities {
	
	// N: the number features
	private static int N = Constants.NO_OF_FEATURES;
		
    // read the feature dictionary
    public static String[] readFeatureDictionary(String featureDictionaryPath) {
    	String[] features = new String[Constants.NO_OF_FEATURES];
		try {
			BufferedReader br = new BufferedReader(new FileReader(new File(featureDictionaryPath)));
			
			String line = null;
			
			int i = 0;
			while ((line = br.readLine()) != null) {
				if (!line.equals("")) {
					features[i] = line;
					i++;
				}
			}
			
			br.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}

    	return features;
	}
    

    // for reading Train data
    public static List<Instance> readData(String dataPath, String[] features, String trainOrTest, String labelString) {
    	
		List<Instance> data = new ArrayList<Instance>();
		int no_of_files = 0;
    	File folder = new File(dataPath);
    	File[] listFiles = folder.listFiles();
    	Arrays.sort(listFiles);
    	for (final File fileEntry : listFiles) {
    		
	        if (!fileEntry.isDirectory()) { // if is file
	        	
	        	// the feature vector for the current file
				int[] intline = new int[N+1];
				
	            System.out.println("Reading " + trainOrTest + " file " + "'" + fileEntry.getName() + "'" + "...");
	        	
	        	// the label of the current file
	    		int label = -1;
	    		
	        	if (labelString.toLowerCase().contains("spam")) {
		        	label = Constants.LABEL_SPAM;
		        } else if (labelString.toLowerCase().contains("ham")) {
	        		label = Constants.LABEL_HAM;
	        	}
	        	
	            no_of_files++;

	    		
	    		String entireFileText = readEntireFileIntoString(dataPath + "/" + fileEntry.getName());
	    		
	    		String[] tokens = entireFileText.split(" ");
				Set<String> tokensSet = new HashSet<String>(Arrays.asList(tokens));

	    		// iterate all the tokens for all the features
	    		// to assign values to the feature vector
	    		intline[0] = 1;  // add bias term

	    		for (int i=0; i<features.length; i++) {
    				String feature = features[i];
	    			if (tokensSet.contains(feature)) {
	    				intline[i] = 1;
	    			}
    			}
	    		
				data.add(new Instance(intline, label));
	    		
	        } // end if
	        
	    } // end folder iteration
    	System.out.println();

    	System.out.println("Number of " + trainOrTest + " " + labelString + " files: " + no_of_files);
    	System.out.println();

		return data;

	}
    
    
    /*** HELPER FUNCTIONS ***/
    
	// read entire file into a String
    private static String readEntireFileIntoString(String file) {
		String entireFileText = "";
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line = null;
			// read entire file into a String
			while ((line = br.readLine()) != null) {
				entireFileText += line + " ";
			}
			br.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return entireFileText;
    }
    
}
