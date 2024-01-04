package first;

import java.io.*;
import java.util.*;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Minor_1_Training {
        private static final Logger log = LoggerFactory
                        .getLogger(Minor_1_Training.class);

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
                int numEpochs = 10;

                // File paths
                File trainData = new File("D:\\Minor_1_Dataset\\marine_species");

                // FileSplit(path, allowed formats, random)
                FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

                ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
                ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
                recordReader.initialize(train);
          
                DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
                // Scale pixel values to 0-1
                DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
                scaler.fit(dataIter);
                dataIter.setPreProcessor(scaler);

                for (int i = 1; i < 3; i++) {
                        DataSet ds = dataIter.next();
                        System.out.println(ds);
                        System.out.println(dataIter.getLabels());
                }

                // CNN
                log.info("Build Model.....");
                 MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(rngseed)
                        .l2(0.0005)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new Adam(1e-3))
                        .list()
                        .layer(new ConvolutionLayer.Builder(5, 5)
                                .nIn(channels)
                                .stride(1,1)
                                .nOut(20)
                                .activation(Activation.IDENTITY)
                                .build())
                        .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                                .kernelSize(2,2)
                                .stride(2,2)
                                .build())
                        .layer(new ConvolutionLayer.Builder(5, 5)
                                .stride(1,1)
                                .nOut(50)
                                .activation(Activation.IDENTITY)
                                .build())
                        .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                                .kernelSize(2,2)
                                .stride(2,2)
                                .build())
                        .layer(new DenseLayer.Builder().activation(Activation.RELU)
                                .nOut(500).build())
                        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(outputNum)
                                .activation(Activation.SOFTMAX)
                                .build())
                        .setInputType(InputType.convolutionalFlat(28,28,1))
                        .build();

                MultiLayerNetwork model = new MultiLayerNetwork(conf);
                model.init();
                model.setListeners(new ScoreIterationListener(10));

                log.info("**** TRAIN MODEL ****");
                for (int i = 0; i < numEpochs; i++) {
                        model.fit(dataIter);
                        log.info("*** Completed epoch {} ****", i);
                }
                model.setListeners(new ScoreIterationListener(100), new EvaluativeListener(dataIter, 1, InvocationType.EPOCH_END)); 
                model.fit(dataIter, numEpochs);
                log.info("***** Saving Trained Model *****");
                File locationToSave =new File("trained_model.zip");
                boolean saveUpdater =false;
                ModelSerializer.writeModel(model, locationToSave, saveUpdater);

                

        }
}
