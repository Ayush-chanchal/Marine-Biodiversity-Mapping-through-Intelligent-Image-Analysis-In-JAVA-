package first;

import javax.swing.JFileChooser;

import java.io.*;
import java.util.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Minor_1_Prediction {
    private static final Logger log = LoggerFactory
                        .getLogger(Minor_1_Prediction.class);
    
    public static String fileChose()
    {
        JFileChooser fc= new JFileChooser();
        int ret = fc.showOpenDialog(null);
        if(ret==JFileChooser.APPROVE_OPTION)
        {
            File file =fc.getSelectedFile();
            String filename = file.getAbsolutePath();
            return filename;           
        }
        else{
            return null;
        }
    }

    public static void main(String args[]) throws Exception
    {
        int height = 28;
        int width = 28;
        int channels = 1;
        List<String> labelList = Arrays.asList("Clams","Corals","Crabs","Dolphin",
                    "Eel","Fish","Jelly Fish","Lobster","Nudibranchs","Octopus","Otter",
                    "Penguin","Puffers","Sea Rays","Sea Urchins","Sea Horse","Seal","Sharks",
                    "Shrimp","Squid","StarFish","Turtle_Tortoise","Whale");
        String filechose= fileChose().toString();
        File locationToSave = new File("trained_model.zip");
        MultiLayerNetwork model=ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        log.info("*** Testing the image:------------");
        File file = new File(filechose);
        NativeImageLoader loader = new NativeImageLoader(height,width,channels);
        INDArray image = loader.asMatrix(file);

        // Normalizing the data in the range of 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.transform(image);

        INDArray output = model.output(image);

        log.info(" The file chosen was "+filechose);
        log.info("Prediction");
        log.info("Probability list");
        log.info("List of labels in orders");
        log.info(output.toString());
        log.info(labelList.toString());
    }
}
