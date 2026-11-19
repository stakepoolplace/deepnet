package RN.transformer;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Généralisation sur des phrases JAMAIS VUES.
 *
 * Tâche : recopie (modèle de langue). Le corpus d'entraînement ne contient
 * aucune des phrases de test, mais tous leurs mots y apparaissent. Un modèle
 * qui a seulement mémorisé des paires échouerait ; un modèle qui a appris le
 * mécanisme (attention encodeur-décodeur) recopie une entrée inédite.
 *
 * LIMITE CONNUE : sur ce micro-modèle (1 couche, 20 phrases) la généralisation
 * est partielle, pas une recopie parfaite (voir seuil ci-dessous).
 *
 * On compare à une base de référence : un modèle non entraîné, pour prouver
 * que le score vient de l'apprentissage.
 */
public class TransformerGeneralizationTest {

    private static final int MAX_LEN = 10;
    private static final int EPOCHS = 150;

    private static final List<String> TRAIN = Arrays.asList(
        "le chat mange", "les chiens aiment", "dans le jardin", "sur le tapis",
        "le chat est dans le jardin", "les chiens aiment manger",
        "le tapis est dans le jardin", "est dans jardin", "le jardin est",
        "chat manger", "chiens aiment", "le chat est sur le tapis",
        "les chiens mangent", "le chat aime le jardin", "chat dans jardin",
        "les chiens sont dans le jardin", "le tapis est sur le chat",
        "le chat aime manger", "dans le tapis", "sur le jardin");

    // Combinaisons inédites de mots déjà vus à l'entraînement.
    private static final List<String> UNSEEN = Arrays.asList(
        "le chat aime le tapis",
        "les chiens mangent dans le jardin",
        "le jardin est sur le tapis",
        "chat sur le jardin",
        "le tapis aime manger",
        "les chiens sont sur le tapis");

    private static String norm(String s) {
        return s == null ? "" : s.replaceAll("<[A-Z]+>", " ").trim().replaceAll("\\s+", " ");
    }

    /** Fraction de mots corrects position par position. */
    private static double tokenAccuracy(String out, String expected) {
        String[] o = out.isEmpty() ? new String[0] : out.split(" ");
        String[] e = expected.split(" ");
        int ok = 0;
        for (int i = 0; i < e.length; i++) {
            if (i < o.length && o[i].equals(e[i])) {
                ok++;
            }
        }
        return ok / (double) e.length;
    }

    private static TransformerModel build(Tokenizer tokenizer, int dim) {
        TransformerModel m = new TransformerModel(1, dim, 4, 512, 0.0, tokenizer.getVocabSize(),
                tokenizer, 0.005f, 2);
        m.setPositionalEncodingScale(0.01);
        return m;
    }

    private static double evaluate(TransformerModel m, String label) {
        double sumAcc = 0;
        int exact = 0;
        for (String s : UNSEEN) {
            String out = norm(m.generate(s, 8, 0.05, 1, 7L));
            double acc = tokenAccuracy(out, s);
            sumAcc += acc;
            if (out.equals(s)) {
                exact++;
            }
            System.out.println("GENERALIZATION " + label + " [" + s + "] -> [" + out + "] acc=" + acc);
        }
        System.out.println("GENERALIZATION " + label + " exact=" + exact + "/" + UNSEEN.size()
                + " tokenAcc=" + (sumAcc / UNSEEN.size()));
        return sumAcc / UNSEEN.size();
    }

    @Test
    public void testGeneralizationOnUnseenSentences() throws Exception {
        Nd4j.getRandom().setSeed(42);
        WordVectors wv = WordVectorSerializer.readWord2VecModel(new File("pretrained-embeddings/mon_model_word2vec.txt"));
        int dim = wv.getWordVector("chat").length;
        Tokenizer tokenizer = new Tokenizer(wv, dim, MAX_LEN);

        // Les phrases de test sont réellement inédites, mais leurs mots sont connus.
        Set<String> trainWords = new HashSet<>();
        for (String t : TRAIN) {
            trainWords.addAll(Arrays.asList(t.split(" ")));
        }
        for (String u : UNSEEN) {
            assertTrue("Phrase de test présente à l'entraînement : " + u, !TRAIN.contains(u));
            for (String w : u.split(" ")) {
                assertTrue("Mot de test absent de l'entraînement : " + w, trainWords.contains(w));
            }
        }

        // Référence : modèle non entraîné (isTrained forcé pour autoriser generate).
        TransformerModel untrained = build(tokenizer, dim);
        untrained.setTrained(true);
        double baseline = evaluate(untrained, "baseline");

        Nd4j.getRandom().setSeed(42);
        TransformerModel model = build(tokenizer, dim);
        DataGenerator gen = new DataGenerator(new ArrayList<>(TRAIN), new ArrayList<>(TRAIN), tokenizer, 4, MAX_LEN);
        model.getOptimizer().setMaxEpochs(EPOCHS);
        float first = -1f, last = 0f;
        for (int e = 0; e < EPOCHS; e++) {
            last = model.train(gen, 1);
            if (e == 0) {
                first = last;
            }
        }
        System.out.println("GENERALIZATION firstLoss=" + first + " lastLoss=" + last);
        assertTrue("L'entraînement doit converger", last < first / 2f);

        double acc = evaluate(model, "trained");
        assertTrue("Le modèle entraîné doit battre le modèle non entraîné (" + acc + " vs " + baseline + ")",
                acc > baseline + 0.3);
        // Mesure observée : ~0,48 (baseline 0,0). Le modèle généralise PARTIELLEMENT
        // (bon début de phrase, mais 0 recopie exacte : arrêts prématurés et
        // substitutions). Seuil volontairement fixé bien au-dessus du hasard
        // (~1/vocab) mais sous la valeur observée, pour détecter une régression
        // sans prétendre à une recopie parfaite.
        assertTrue("Précision par mot sur phrases inédites insuffisante : " + acc, acc >= 0.3);
    }
}
