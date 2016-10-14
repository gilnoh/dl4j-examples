package org.deeplearning4j.examples.nlp.word2vec;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.UiServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collection;

/**
 * Created by agibsonccc on 10/9/14.
 */
public class Word2VecRawTextExample {

    // Modified to generate Brille24 data.
    private static Logger log = LoggerFactory.getLogger(Word2VecRawTextExample.class);

    public static void main(String[] args) throws Exception {

//        String filePath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();
        String filePath = "/home/tailblues/omq/brille24_11krequest.txt";

        log.info("Load & Vectorize Sentences....");
        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(filePath);
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .elementsLearningAlgorithm(new SkipGram<VocabWord>())
//                .elementsLearningAlgorithm(new CBOW<VocabWord>())
                .minWordFrequency(5)
                .iterations(1)
                .epochs(10)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Writing word vectors to text file....");

        // Write word vectors
        WordVectorSerializer.writeWordVectors(vec, "Brille24Vectors.txt");

//        WordVectors vec = WordVectorSerializer.loadTxtVectors(new File("Brille24Vectors_300dim.txt"));
//        WordVectors vec = WordVectorSerializer.loadTxtVectors(new File("/home/tailblues/Dropbox/Brille24Vectors.txt"));

        System.out.println(vec.wordsNearest("brille", 20));
        System.out.println(vec.wordsNearest("paypal", 20));
        System.out.println(vec.wordsNearest("versand", 20));
        System.out.println(vec.wordsNearest("sendung", 20));
        System.out.println(vec.wordsNearest("lieferung", 20));
        System.out.println(vec.wordsNearest("juni", 20));

        System.out.println(vec.similarity("paypal", "bezahlung"));
        System.out.println(vec.similarity("lieferung", "sendung"));
        System.out.println(vec.similarity("paket", "lieferung"));
        System.out.println(vec.similarity("paket", "versand"));

//        UiServer server = UiServer.getInstance();
//        System.out.println("Started on port " + server.getPort());
    }
}
