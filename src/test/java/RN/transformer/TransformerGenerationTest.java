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
 * Entraînement sur un corpus diversifié (phrases complètes, fragments,
 * insertions, pluriels) puis génération libre de texte par échantillonnage
 * (température + top-k) au lieu de l'argmax glouton qui fige le modèle sur
 * un token répété ("le le le...").
 *
 * Config alignée sur les implémentations modernes pour un micro-corpus :
 * 1 couche (la profondeur Post-LN piège l'optimisation sur un attracteur
 * marginal : sorties identiques quelle que soit l'amorce), embeddings
 * × sqrt(dModel) + EOS supervisé + START/END entraînables (cf. correctifs
 * TransformerModel/Tokenizer). Le test prouve le CONDITIONNEMENT : des
 * amorces différentes produisent des sorties différentes.
 */
public class TransformerGenerationTest {

    @Test
    public void testDiverseCorpusGeneration() throws Exception {
        Nd4j.getRandom().setSeed(42);
        int maxSequenceLength = 10;
        WordVectors preTrainedWordVectors = org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
                .readWord2VecModel(new File("pretrained-embeddings/mon_model_word2vec.txt"));
        int embeddingSize = preTrainedWordVectors.getWordVector("chat").length;
        Tokenizer tokenizer = new Tokenizer(preTrainedWordVectors, embeddingSize, maxSequenceLength);

        // Corpus diversifié (tous les mots sont dans le vocabulaire word2vec) :
        // phrases complètes, fragments, une insertion et un pluriel.
        List<String> inputs = Arrays.asList(
            "le chat mange",
            "les chiens aiment",
            "chat sur tapis",
            "dans le jardin",
            "le chat est dans le jardin",
            "sur le tapis",
            "les chiens aiment manger",
            "le tapis est dans le jardin",
            "est dans jardin",
            "le jardin est",
            "chat manger",
            "chiens aiment"
        );
        // Cible = recopie (modèle de langue) sauf insertion explicite.
        List<String> targets = Arrays.asList(
            "le chat mange",
            "les chiens aiment",
            "chat est sur tapis",
            "dans le jardin",
            "le chat est dans le jardin",
            "sur le tapis",
            "les chiens aiment manger",
            "le tapis est dans le jardin",
            "est dans jardin",
            "le jardin est",
            "chat manger",
            "chiens aiment"
        );

        int numLayers = 1; // Micro-corpus : 1 couche suffit et conditionne ;
                            // 2+ couches Post-LN restent bloquées sur l'attracteur
                            // marginal (même sortie quelle que soit l'amorce).
        int dModel = embeddingSize;
        int numHeads = 4;
        int dff = 512;
        float dropoutRate = 0.0f;
        float initialLr = 0.005f;
        int warmupSteps = 2;
        TransformerModel model = new TransformerModel(numLayers, dModel, numHeads, dff,
                dropoutRate, tokenizer.getVocabSize(), tokenizer, initialLr, warmupSteps);
        model.setPositionalEncodingScale(0.01);

        DataGenerator gen = new DataGenerator(inputs, targets, tokenizer, 4, maxSequenceLength);
        int epochs = 80;
        model.getOptimizer().setMaxEpochs(epochs);
        float firstLoss = -1f;
        float lastLoss = 0f;
        for (int e = 0; e < epochs; e++) {
            lastLoss = model.train(gen, 1);
            if (e == 0) {
                firstLoss = lastLoss;
            }
            if ((e + 1) % 20 == 0) {
                System.out.println("GENERATION epoch " + (e + 1) + " loss=" + lastLoss);
            }
        }
        System.out.println("GENERATION firstLoss=" + firstLoss + " lastLoss=" + lastLoss);
        assertTrue("L'entraînement devrait faire baisser la perte (init=" + firstLoss + ", finale=" + lastLoss + ")",
                lastLoss < firstLoss);

        // Génération libre (température + top-k) sur des amorces variées.
        String[] prompts = { "le chat", "les chiens", "dans le", "chat sur", "le jardin est", "est dans" };
        int nonEmpty = 0;
        java.util.Set<String> distinctOutputs = new java.util.HashSet<>();
        for (String prompt : prompts) {
            String out = model.generate(prompt, 6, 0.9, 6, 12345L);
            System.out.println("GENERATION prompt=[" + prompt + "] -> [" + out + "]");
            assertNotNull("La génération ne devrait pas être null", out);
            if (!out.trim().isEmpty()) {
                nonEmpty++;
                distinctOutputs.add(out.trim());
            }
            assertTrue("La génération ne devrait pas contenir de <UNK> (obtenu: '" + out + "')",
                    !out.contains("<UNK>"));
        }
        System.out.println("GENERATION prompts non vides : " + nonEmpty + "/" + prompts.length
                + ", sorties distinctes : " + distinctOutputs.size());
        assertTrue("Au moins la moitié des amorces devraient produire du texte", nonEmpty * 2 >= prompts.length);
        // CONDITIONNEMENT : le modèle répond à l'amorce (fini le "même texte
        // quelle que soit l'entrée"). Au moins 2 sorties distinctes.
        assertTrue("Le modèle devrait conditionner ses sorties sur l'amorce (obtenu une seule sortie : "
                + distinctOutputs + ")", distinctOutputs.size() >= 2);
    }
}
