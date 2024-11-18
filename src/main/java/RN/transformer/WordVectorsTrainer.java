package RN.transformer;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.List;

public class WordVectorsTrainer {
    public static Word2Vec trainWordVectors() {
        List<String> sentences = Arrays.asList(
            "<START> chat mange la souris <END>",
            "<START> chien court dans le jardin <END>",
            "<START> les chats aiment le tapis sur le sol <END>"
        );

        SentenceIterator iter = new CollectionSentenceIterator(sentences);
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(1)
                .iterations(1)
                .layerSize(3) // Petite taille pour les tests
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(new DefaultTokenizerFactory())
                .build();
        vec.fit();
        return vec;
    }

    public static void main(String[] args) {
        Word2Vec word2Vec = trainWordVectors();
        // Sauvegarder le modèle si nécessaire
        WordVectorSerializer.writeWord2VecModel(word2Vec, "src/test/resources/word2vec/test_word2vec_model.txt");
    }
}
