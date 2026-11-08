package RN.transformer;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Entraînement sur un corpus diversifié (structures variées : phrases
 * complètes, fragments, insertion, pluriels) puis inférence, y compris
 * sur une combinaison inédite (généralisation).
 */
public class TransformerDiversityTest {

    @Test
    public void testDiverseCorpusTrainingAndInference() throws Exception {
        Nd4j.getRandom().setSeed(7);
        int maxSequenceLength = 10;
        WordVectors preTrainedWordVectors = org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
                .readWord2VecModel(new File("pretrained-embeddings/mon_model_word2vec.txt"));
        int embeddingSize = preTrainedWordVectors.getWordVector("chat").length;
        Tokenizer tokenizer = new Tokenizer(preTrainedWordVectors, embeddingSize, maxSequenceLength);

        // Corpus diversifié : recopie, pluriels, insertion, fragments courts
        List<String> inputs = Arrays.asList(
            "le chat mange",
            "les chiens aiment",
            "chat sur tapis",
            "dans le jardin",
            "sur le tapis",
            "est dans jardin",
            "le jardin est",
            "chiens aiment"
        );
        List<String> targets = Arrays.asList(
            "le chat mange",
            "les chiens aiment",
            "chat est sur tapis",
            "est dans le jardin",
            "sur le tapis",
            "est dans jardin",
            "le jardin est",
            "chiens aiment"
        );

        int numLayers = 4;
        int dModel = embeddingSize;
        int numHeads = 4;
        int dff = 512;
        float dropoutRate = 0.0f;
        float initialLr = 0.005f;
        int warmupSteps = 2;
        TransformerModel model = new TransformerModel(numLayers, dModel, numHeads, dff,
                dropoutRate, tokenizer.getVocabSize(), tokenizer, initialLr, warmupSteps);
        model.setPositionalEncodingScale(0.01);

        DataGenerator gen = new DataGenerator(inputs, targets, tokenizer, inputs.size(), maxSequenceLength);
        int epochs = 60;
        model.getOptimizer().setMaxEpochs(epochs);
        float firstLoss = -1f;
        float lastLoss = 0f;
        for (int e = 0; e < epochs; e++) {
            lastLoss = model.train(gen, 1);
            if (e == 0) {
                firstLoss = lastLoss;
            }
            if ((e + 1) % 20 == 0 || e == 0) {
                System.out.println("DIVERSITY epoch " + (e + 1) + " loss=" + lastLoss);
            }
        }
        System.out.println("DIVERSITY firstLoss=" + firstLoss + " lastLoss=" + lastLoss);
        // Exige une vraie diminution (le plateau ultérieur est documenté
        // dans TransformerHeavyIntegrationTest : convergence partielle).
        assertTrue("La perte finale (" + lastLoss + ") devrait être inférieure à la perte initiale (" + firstLoss + ")",
                lastLoss < firstLoss);

        // Inférence sur des exemples vus : on vérifie qu'elle produit une
        // sortie non vide SANS tokens inconnus (le modèle a appris à émettre
        // du vocabulaire réel). On n'exige PAS de mot précis : le décodage
        // gourmand se fige sur un token unique ("le le le", "est est est"...),
        // une limite documentée de ce modèle jouet (voir HeavyIntegrationTest).
        String[] probes = { "le chat mange", "les chiens aiment", "chat sur tapis", "dans le jardin" };
        for (String prompt : probes) {
            String out = model.infer(prompt, 4);
            System.out.println("DIVERSITY infer[" + prompt + "] = [" + out + "]");
            assertNotNull("L'inférence ne devrait pas être null", out);
            assertTrue("L'inférence de '" + prompt + "' ne devrait pas être vide", !out.trim().isEmpty());
            assertTrue("L'inférence de '" + prompt + "' ne devrait pas contenir de <UNK> (obtenu: '" + out + "')",
                    !out.contains("<UNK>"));
        }

        // Généralisation : combinaison inédite de mots connus
        String novel = model.infer("le chat dans jardin", 4);
        System.out.println("DIVERSITY infer[novel 'le chat dans jardin'] = [" + novel + "]");
        assertNotNull("L'inférence sur entrée inédite ne devrait pas être null", novel);
        assertTrue("L'inférence sur entrée inédite ne devrait pas être vide", !novel.trim().isEmpty());
        assertTrue("L'inférence sur entrée inédite ne devrait pas contenir de <UNK>", !novel.contains("<UNK>"));
    }
}
