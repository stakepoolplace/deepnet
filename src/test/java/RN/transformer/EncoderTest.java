package RN.transformer;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import RN.utils.NDArrayUtils;

public class EncoderTest {

    @Test
    public void testMask() {

        // Initialisation du Tokenizer
        List<String> vocab = Arrays.asList("<PAD>", "<START>", "<END>", "hello", "world");
        Tokenizer tokenizer = new Tokenizer(vocab, 300, 11);

        int padTokenId = tokenizer.getPadTokenId(); // Supposons que l'ID du token <PAD> est 0
        System.out.println("Pad token id:\n" + padTokenId);

        // Séquences avec paddings à différentes positions
        INDArray tokenIds1 = Nd4j.create(new float[][] {
                { 1, 2, 3, 0, 0 }
        });

        INDArray tokenIds2 = Nd4j.create(new float[][] {
                { 0, 1, 2, 3, 0 }
        });

        INDArray tokenIds3 = Nd4j.create(new float[][] {
                { 1, 0, 2, 0, 3 }
        });

        INDArray tokenIds4 = Nd4j.create(new float[][] {
                { 1 }
        });

        // Générer et afficher les masques
        INDArray keyMask1 = NDArrayUtils.createKeyPaddingMask(tokenizer, tokenIds1);
        INDArray queryMask1 = NDArrayUtils.createQueryPaddingMask(tokenizer, tokenIds1);
        System.out.println("Key Mask 1:\n" + keyMask1);
        System.out.println("Query Mask 1:\n" + queryMask1);

        INDArray keyMask2 = NDArrayUtils.createKeyPaddingMask(tokenizer, tokenIds2);
        INDArray queryMask2 = NDArrayUtils.createQueryPaddingMask(tokenizer, tokenIds2);
        System.out.println("Key Mask 2:\n" + keyMask2);
        System.out.println("Query Mask 2:\n" + queryMask2);

        INDArray keyMask3 = NDArrayUtils.createKeyPaddingMask(tokenizer, tokenIds3);
        INDArray queryMask3 = NDArrayUtils.createQueryPaddingMask(tokenizer, tokenIds3);
        System.out.println("Key Mask 3:\n" + keyMask3);
        System.out.println("Query Mask 3:\n" + queryMask3);

        INDArray keyMask4 = NDArrayUtils.createKeyPaddingMask(tokenizer, tokenIds4);
        INDArray queryMask4 = NDArrayUtils.createQueryPaddingMask(tokenizer, tokenIds4);
        System.out.println("Key Mask 4:\n" + keyMask4);
        System.out.println("Query Mask 4:\n" + queryMask4);
    }

    @Test
    public void testQKVFictifs() {
        // Initialisation du Tokenizer
        List<String> vocab = Arrays.asList("<PAD>", "<START>", "<END>", "chat", "mange", "souris", "la");
        Tokenizer tokenizer = new Tokenizer(vocab, 300, 11);

        // Exemple de tokens avec padding à la fin
        // Séquence : [<START>, chat, mange, la, souris, <END>, <PAD>, <PAD>, <PAD>]
        INDArray tokenIds = Nd4j.create(new float[][] {
                { 1, 2, 3, 4, 5, 6, 0, 0, 0 } // batchSize=1, seqLength=9
        });

        // Création des masques
        INDArray keyMask = NDArrayUtils.createKeyPaddingMask(tokenizer, tokenIds); // [1, 1, 1, 9]
        INDArray queryMask = NDArrayUtils.createQueryPaddingMask(tokenizer, tokenIds); // [1, 1, 9, 1]

        // Initialisation de MultiHeadAttention
        MultiHeadAttention mha = new MultiHeadAttention(512, 8); // Exemple avec dModel=512, numHeads=8

        // Création de Q, K, V (embeddings fictifs pour le test)
        INDArray query = Nd4j.randn(1, 9, 512); // [batchSize, seqLength_q, dModel]
        INDArray key = Nd4j.randn(1, 9, 512); // [batchSize, seqLength_k, dModel]
        INDArray value = Nd4j.randn(1, 9, 512); // [batchSize, seqLength_k, dModel]

        // Application de la passe forward
        INDArray output = mha.forward(query, key, value, queryMask, keyMask, null);

        // Inspection des poids d'attention pour les requêtes <PAD>
        mha.printAttentionWeights(
                Arrays.asList("<START>", "chat", "mange", "la", "souris", "<END>", "<PAD>", "<PAD>", "<PAD>"),
                Arrays.asList("<START>", "chat", "mange", "la", "souris", "<END>", "<PAD>", "<PAD>", "<PAD>"),
                0,
                null);

        // Vérifiez que les poids d'attention pour les requêtes <PAD> sont tous à 0
    }

}
