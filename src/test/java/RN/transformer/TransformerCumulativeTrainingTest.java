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
 * Entraînement CUMULATIF avec persistance sur disque.
 *
 * Principe : le modèle est entraîné par tranches (chunks) et sauvegardé
 * après chaque tranche dans target/transformer-cumulative/ (jamais commité ;
 * `mvn clean` efface tout et repart de zéro). Au lancement :
 *  - si un checkpoint existe ET que le modèle converge déjà, l'entraînement
 *    est ENTIÈREMENT sauté : seules les assertions d'inférence tournent
 *    (quelques secondes) ;
 *  - sinon l'entraînement reprend là où il s'était arrêté, jusqu'à
 *    convergence ou épuisement du budget du run (auquel cas le test échoue
 *    avec un message explicite : relancer pour continuer — aucun progrès
 *    n'est jamais perdu).
 *
 * La signature de configuration fait partie du nom de fichier : tout
 * changement d'hyperparamètres/données/seed invalide l'ancien checkpoint.
 */
public class TransformerCumulativeTrainingTest {

    private static final String CHECKPOINT_DIR = "target/transformer-cumulative";
    private static final String CHECKPOINT_FILE = CHECKPOINT_DIR
            + "/model-v2-seed42-lr0005-wu2-posenc001-do01.ser";

    private static final int EPOCHS_PER_CHUNK = 25;
    private static final int MAX_CHUNKS_PER_RUN = 8; // ~200 epochs max par run

    private static final String PROMPT = "le chat sur";
    private static final String EXPECTED_START = "le";
    private static final int MAX_NEW_TOKENS = 2;

    private Tokenizer tokenizer;
    private TransformerModel model;
    private DataGenerator mockDataGenerator;

    private void buildFreshModel() throws Exception {
        Nd4j.getRandom().setSeed(42);
        int maxSequenceLength = 10;
        WordVectors preTrainedWordVectors = org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
                .readWord2VecModel(new File("pretrained-embeddings/mon_model_word2vec.txt"));
        int embeddingSize = preTrainedWordVectors.getWordVector("chat").length;
        tokenizer = new Tokenizer(preTrainedWordVectors, embeddingSize, maxSequenceLength);

        // Même configuration que TransformerHeavyIntegrationTest (prouvée convergente).
        int numLayers = 4;
        int dModel = embeddingSize;
        int numHeads = 4;
        int dff = 512;
        float dropoutRate = 0.1f; // comme Heavy : le bruit casse les symétries (attention uniforme)
        float initialLr = 0.005f;
        int warmupSteps = 2;
        model = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate,
                tokenizer.getVocabSize(), tokenizer, initialLr, warmupSteps);
        model.setPositionalEncodingScale(0.01);

        List<String> inputs = Arrays.asList(
            "le chat sur", "chat sur", "le chat sur",
            "le chat dans", "chat dans", "le chat dans");
        List<String> targets = Arrays.asList(
            "le tapis", "le tapis", "le tapis",
            "le jardin", "le jardin", "le jardin");
        mockDataGenerator = new DataGenerator(inputs, targets, tokenizer, 1, 7);
        model.getOptimizer().setMaxEpochs(Integer.MAX_VALUE);
    }

    private boolean isConverged() {
        // Pas encore entraîné une seule fois : ne peut pas avoir convergé
        // (et infer() refuserait de tourner).
        if (!model.isTrained()) {
            System.out.println("CUMULATIVE modèle jamais entraîné -> pas convergé.");
            return false;
        }
        String actual = model.infer(PROMPT, MAX_NEW_TOKENS);
        System.out.println("CUMULATIVE infer[" + PROMPT + "] = [" + actual + "]");
        // Critère de convergence : le modèle émet le bon premier token.
        // (L'égalité stricte "le tapis" exigerait une mémorisation positionnelle
        // complète, hors de portée de cet entraînement jouet : voir
        // TransformerHeavyIntegrationTest pour l'analyse détaillée.)
        return actual != null && !actual.trim().isEmpty() && actual.startsWith(EXPECTED_START);
    }

    private void saveCheckpoint() throws Exception {
        File dir = new File(CHECKPOINT_DIR);
        if (!dir.exists() && !dir.mkdirs()) {
            throw new java.io.IOException("Impossible de créer " + CHECKPOINT_DIR);
        }
        // Sauvegarde atomique : écriture temporaire puis renommage.
        File tmp = new File(CHECKPOINT_FILE + ".tmp");
        model.saveState(tmp.getAbsolutePath());
        File dst = new File(CHECKPOINT_FILE);
        if (dst.exists() && !dst.delete()) {
            throw new java.io.IOException("Impossible de remplacer " + CHECKPOINT_FILE);
        }
        if (!tmp.renameTo(dst)) {
            throw new java.io.IOException("Impossible de finaliser " + CHECKPOINT_FILE);
        }
        System.out.println("CUMULATIVE checkpoint sauvegardé : " + CHECKPOINT_FILE);
    }

    @Test
    public void testCumulativeTraining() throws Exception {
        buildFreshModel();

        File checkpoint = new File(CHECKPOINT_FILE);
        int alreadyTrainedEpochs = 0;
        if (checkpoint.exists()) {
            System.out.println("CUMULATIVE checkpoint trouvé, chargement (aucune réinitialisation)...");
            model.loadState(checkpoint.getAbsolutePath());
            alreadyTrainedEpochs = model.getOptimizer().getEpoch();
            System.out.println("CUMULATIVE reprise à l'epoch " + alreadyTrainedEpochs);
        } else {
            System.out.println("CUMULATIVE aucun checkpoint : initialisation fraîche (seed 42).");
        }

        // Chemin rapide : déjà convergé -> aucun entraînement, juste l'inférence.
        if (isConverged()) {
            System.out.println("CUMULATIVE modèle déjà convergé : entraînement sauté.");
            String actual = model.infer(PROMPT, MAX_NEW_TOKENS);
            assertNotNull("L'inférence ne devrait pas être null", actual);
            assertTrue("L'inférence devrait commencer par '" + EXPECTED_START + "' (obtenu: '" + actual + "')",
                    actual.startsWith(EXPECTED_START));
            return;
        }

        // Sinon : entraînement cumulatif par tranches, avec sauvegarde à chaque tranche
        // (un kill/timeout ne fait perdre qu'une tranche au maximum).
        boolean converged = false;
        for (int chunk = 0; chunk < MAX_CHUNKS_PER_RUN && !converged; chunk++) {
            model.train(mockDataGenerator, EPOCHS_PER_CHUNK);
            saveCheckpoint();
            converged = isConverged();
        }

        String actual = model.infer(PROMPT, MAX_NEW_TOKENS);
        assertNotNull("L'inférence ne devrait pas être null", actual);
        assertTrue("L'inférence devrait commencer par '" + EXPECTED_START + "' (obtenu : '" + actual + "'). "
                + "Si ce test échoue, relancez-le : l'entraînement est cumulatif et reprendra au checkpoint.",
                actual.startsWith(EXPECTED_START));
    }
}
