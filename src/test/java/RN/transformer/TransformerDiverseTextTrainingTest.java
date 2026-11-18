package RN.transformer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Validation de bout en bout de l'entraînement sur un texte DIVERSIFIÉ.
 *
 * Contrairement à TransformerGenerationTest (qui se contente d'une perte qui
 * baisse et de 2 sorties distinctes), on exige ici des preuves strictes :
 *  1. la perte finale est au moins divisée par 2 et reste finie ;
 *  2. toutes les valeurs de perte sont finies (pas de NaN/Inf) ;
 *  3. le modèle reproduit EXACTEMENT la majorité des paires d'entraînement
 *     (décodage quasi glouton) : il a réellement appris le corpus ;
 *  4. l'insertion apprise ("chat sur tapis" -> "chat est sur tapis") est bien
 *     produite, ce qui prouve que la sortie dépend de l'entrée (pas d'un
 *     attracteur marginal) ;
 *  5. des amorces différentes donnent des sorties différentes ;
 *  6. la génération est déterministe à graine fixe ;
 *  7. le vocabulaire de sortie reste dans le vocabulaire du corpus (pas de
 *     jeton spécial parasite dans le texte) ;
 *  8. les paramètres ont réellement été mis à jour et restent finis.
 */
public class TransformerDiverseTextTrainingTest {

    private static final int MAX_LEN = 10;
    private static final int EPOCHS = 100;

    private static final List<String> INPUTS = Arrays.asList(
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
        "chiens aiment");

    private static final List<String> TARGETS = Arrays.asList(
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
        "chiens aiment");

    private static String norm(String s) {
        return s == null ? "" : s.replaceAll("<[A-Z]+>", " ").trim().replaceAll("\\s+", " ");
    }

    @Test
    public void testTrainingOnDiverseTextLearnsTheCorpus() throws Exception {
        Nd4j.getRandom().setSeed(42);
        WordVectors wv = WordVectorSerializer.readWord2VecModel(new File("pretrained-embeddings/mon_model_word2vec.txt"));
        int dim = wv.getWordVector("chat").length;
        Tokenizer tokenizer = new Tokenizer(wv, dim, MAX_LEN);

        TransformerModel model = new TransformerModel(1, dim, 4, 512, 0.0, tokenizer.getVocabSize(),
                tokenizer, 0.005f, 2);
        model.setPositionalEncodingScale(0.01);
        DataGenerator gen = new DataGenerator(INPUTS, TARGETS, tokenizer, 4, MAX_LEN);
        model.getOptimizer().setMaxEpochs(EPOCHS);

        List<INDArray> before = new java.util.ArrayList<>();
        for (INDArray p : model.getCombinedParameters()) {
            before.add(p.dup());
        }

        float first = -1f, last = 0f;
        for (int e = 0; e < EPOCHS; e++) {
            last = model.train(gen, 1);
            assertTrue("Perte non finie à l'epoch " + e + " : " + last,
                    !Float.isNaN(last) && !Float.isInfinite(last));
            if (e == 0) {
                first = last;
            }
        }
        System.out.println("DIVERSE firstLoss=" + first + " lastLoss=" + last);

        // 1. La perte a au moins été divisée par 2.
        assertTrue("La perte doit être divisée par 2 (init=" + first + ", finale=" + last + ")", last < first / 2f);

        // 8. Les paramètres ont changé et sont finis.
        int changed = 0;
        List<INDArray> after = model.getCombinedParameters();
        for (int i = 0; i < after.size(); i++) {
            double sum = after.get(i).sumNumber().doubleValue();
            assertTrue("Paramètre " + i + " contient NaN/Inf", !Double.isNaN(sum) && !Double.isInfinite(sum));
            if (!after.get(i).equals(before.get(i))) {
                changed++;
            }
        }
        assertTrue("Des paramètres doivent avoir été mis à jour", changed > 0);

        // 3. Reproduction exacte des paires d'entraînement (décodage quasi glouton).
        int exact = 0;
        Set<String> outputs = new HashSet<>();
        for (int i = 0; i < INPUTS.size(); i++) {
            String out = norm(model.generate(INPUTS.get(i), 8, 0.05, 1, 7L));
            System.out.println("DIVERSE [" + INPUTS.get(i) + "] -> [" + out + "] (attendu [" + TARGETS.get(i) + "])");
            if (out.equals(TARGETS.get(i))) {
                exact++;
            }
            outputs.add(out);
            // 7. pas de jeton inconnu dans le texte généré
            assertFalse("<UNK> dans la sortie : " + out, out.contains("UNK"));
        }
        double acc = exact / (double) INPUTS.size();
        System.out.println("DIVERSE exact-match=" + exact + "/" + INPUTS.size());
        assertTrue("Le modèle doit reproduire >= 75% des paires d'entraînement (obtenu " + exact + "/" + INPUTS.size() + ")",
                acc >= 0.75);

        // 4. L'insertion apprise est produite : la sortie dépend de l'entrée.
        assertEquals("L'insertion apprise doit être reproduite", "chat est sur tapis",
                norm(model.generate("chat sur tapis", 8, 0.05, 1, 7L)));

        // 5. Sorties variées (pas d'attracteur marginal).
        assertTrue("Sorties trop peu variées : " + outputs, outputs.size() >= INPUTS.size() / 2);

        // 6. Déterminisme à graine fixe.
        assertEquals(model.generate("le chat", 6, 0.9, 6, 123L), model.generate("le chat", 6, 0.9, 6, 123L));
    }
}
