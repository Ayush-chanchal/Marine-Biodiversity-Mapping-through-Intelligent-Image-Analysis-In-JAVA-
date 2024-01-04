package first;

import java.io.*;
import java.util.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.eval.*;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Minor_Evaluation_Test {
        private static final Logger log = LoggerFactory
                        .getLogger(Minor_Evaluation_Test.class);

        public static void main(String[] args) throws IOException {
                // image info
                // 28 X 28 grayscale
                // grayscale implies single channel
                int height = 28;
                int width = 28;
                int channels = 1;
                int rngseed = 123;
                Random randNumGen = new Random(rngseed);
                int batchSize = 128;
                int outputNum = 23;

                // File paths
                File trainData = new File("D:\\Minor_1_Dataset\\marine_species");
                File testData = new File("D:\\Minor_1_Dataset\\Marine_species_Dataset\\Testing");

                // FileSplit(path, allowed formats, random)
                FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
                FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

                ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
                ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
                recordReader.initialize(train);
          
                DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
                // Scale pixel values to 0-1
                DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
                scaler.fit(dataIter);
                dataIter.setPreProcessor(scaler);

                log.info("***** Loading The model *****");
                File locationToSave = new File("trained_model.zip");
                MultiLayerNetwork model=ModelSerializer.restoreMultiLayerNetwork(locationToSave);
                model.getLabels();

                log.info("***** Evaluate Model *****");
                recordReader.initialize(test);
                DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
                scaler.fit(testIter);
                testIter.setPreProcessor(scaler);
                Evaluation eval = new Evaluation(outputNum);

                while (testIter.hasNext()) {
                        DataSet next = testIter.next();
                        INDArray output = model.output(next.getFeatures(), false);
                        eval.eval(next.getLabels(), output);
                }
                log.info(eval.stats());// to get the Accuracy
                log.info("Accuracy: {}", eval.accuracy()*100+"%");
        }
}