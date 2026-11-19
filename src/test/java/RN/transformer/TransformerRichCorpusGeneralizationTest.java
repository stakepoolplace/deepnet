package RN.transformer;

import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Généralisation sur un corpus RICHE généré par grammaire.
 *
 * Contrairement à TransformerGeneralizationTest (limité aux 11 mots du fichier
 * word2vec fourni, dont 4 n'y figurent même pas), ce test construit son propre
 * vocabulaire (~60 mots : déterminants, noms, adjectifs, verbes, prépositions)
 * et génère des centaines de phrases de structures et longueurs variées.
 * Les phrases de test sont tirées de la MÊME grammaire mais ne figurent jamais
 * dans l'entraînement ; tous leurs mots y sont vus.
 *
 * Tâche : recopie (modèle de langue). Les embeddings sont appris
 * (initialisation aléatoire), pas pré-entraînés.
 */
public class TransformerRichCorpusGeneralizationTest {

    private static final int MAX_LEN = 12;
    private static final int D_MODEL = 64;
    private static final int TARGET_EPOCHS = 120;   // total cumulé visé
    private static final int EPOCHS_PER_RUN = 20;   // budget d'un lancement (reprise au suivant)
    private static final float LR = 0.001f;
    private static final String CKPT_DIR = "target/transformer-rich";
    private static final String CKPT = CKPT_DIR + "/model-v1-lr0001-seed42.ser";
    private static final String CKPT_EPOCHS = CKPT + ".epochs";

    private static final String[] DET = { "le", "la", "les", "un", "une" };
    private static final String[] NOUN = { "chat", "chien", "oiseau", "enfant", "livre", "table", "jardin",
            "maison", "voiture", "arbre", "riviere", "montagne", "ville", "route", "cheval" };
    private static final String[] ADJ = { "petit", "grand", "rouge", "vert", "vieux", "joli", "calme", "rapide" };
    private static final String[] VERB = { "voit", "aime", "cherche", "trouve", "garde", "suit", "regarde",
            "dessine" };
    private static final String[] PREP = { "dans", "sur", "pres", "devant", "derriere", "avec" };

    private static String pick(Random r, String[] a) {
        return a[r.nextInt(a.length)];
    }

    /** DET [ADJ] NOUN VERB [PREP] DET [ADJ] NOUN  (5 à 8 mots). */
    private static String sentence(Random r) {
        StringBuilder sb = new StringBuilder();
        sb.append(pick(r, DET)).append(' ');
        if (r.nextBoolean()) {
            sb.append(pick(r, ADJ)).append(' ');
        }
        sb.append(pick(r, NOUN)).append(' ').append(pick(r, VERB)).append(' ');
        if (r.nextBoolean()) {
            sb.append(pick(r, PREP)).append(' ');
        }
        sb.append(pick(r, DET)).append(' ');
        if (r.nextBoolean()) {
            sb.append(pick(r, ADJ)).append(' ');
        }
        sb.append(pick(r, NOUN));
        return sb.toString();
    }

    private static String norm(String s) {
        return s == null ? "" : s.replaceAll("<[A-Z]+>", " ").trim().replaceAll("\\s+", " ");
    }

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

    private static double evaluate(TransformerModel m, List<String> sentences, String label) {
        double sum = 0;
        int exact = 0;
        for (String s : sentences) {
            String out = norm(m.generate(s, 10, 0.05, 1, 7L));
            sum += tokenAccuracy(out, s);
            if (out.equals(s)) {
                exact++;
            }
        }
        System.out.println("RICH " + label + " exact=" + exact + "/" + sentences.size()
                + " tokenAcc=" + (sum / sentences.size()));
        return sum / sentences.size();
    }

    @Test
    public void testGeneralizationOnRichGeneratedCorpus() throws Exception {
        Nd4j.getRandom().setSeed(42);
        Random rnd = new Random(2024);

        Set<String> all = new LinkedHashSet<>();
        while (all.size() < 400) {
            all.add(sentence(rnd));
        }
        List<String> pool = new ArrayList<>(all);
        List<String> test = new ArrayList<>(pool.subList(0, 30));
        List<String> train = new ArrayList<>(pool.subList(30, pool.size()));

        Set<String> vocabWords = new LinkedHashSet<>();
        for (String[] a : new String[][] { DET, NOUN, ADJ, VERB, PREP }) {
            vocabWords.addAll(Arrays.asList(a));
        }
        List<String> vocabulary = new ArrayList<>(Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>"));
        vocabulary.addAll(vocabWords);

        // Garanties du protocole : test inédit, mots tous vus à l'entraînement.
        Set<String> seen = new LinkedHashSet<>();
        for (String t : train) {
            seen.addAll(Arrays.asList(t.split(" ")));
        }
        for (String t : test) {
            assertTrue("Phrase de test présente à l'entraînement : " + t, !train.contains(t));
            for (String w : t.split(" ")) {
                assertTrue("Mot de test jamais vu à l'entraînement : " + w, seen.contains(w));
                assertTrue("Mot hors vocabulaire : " + w, vocabulary.contains(w));
            }
        }
        System.out.println("RICH vocab=" + vocabulary.size() + " train=" + train.size() + " test=" + test.size());

        Tokenizer tokenizer = new Tokenizer(vocabulary, D_MODEL, MAX_LEN);
        TransformerModel untrained = new TransformerModel(2, D_MODEL, 4, 256, 0.0, tokenizer.getVocabSize(),
                tokenizer, LR, 50);
        untrained.setTrained(true);
        double baseline = evaluate(untrained, test, "baseline");

        Nd4j.getRandom().setSeed(42);
        TransformerModel model = new TransformerModel(2, D_MODEL, 4, 256, 0.0, tokenizer.getVocabSize(),
                tokenizer, LR, 50);
        model.getOptimizer().setMaxEpochs(Integer.MAX_VALUE);
        int done = 0;
        java.io.File ck = new java.io.File(CKPT);
        java.io.File ce = new java.io.File(CKPT_EPOCHS);
        if (ck.exists() && ce.exists()) {
            model.loadState(CKPT);
            done = Integer.parseInt(new String(java.nio.file.Files.readAllBytes(ce.toPath())).trim());
            System.out.println("RICH reprise du checkpoint : " + done + " epochs déjà faits");
        }
        DataGenerator gen = new DataGenerator(train, new ArrayList<>(train), tokenizer, 16, MAX_LEN);
        float first = -1f, last = Float.NaN;
        int budget = Math.min(EPOCHS_PER_RUN, TARGET_EPOCHS - done);
        for (int e = 0; e < budget; e++) {
            last = model.train(gen, 1);
            assertTrue("Perte non finie : " + last, !Float.isNaN(last) && !Float.isInfinite(last));
            if (first < 0) {
                first = last;
            }
            done++;
            System.out.println("RICH epoch " + done + "/" + TARGET_EPOCHS + " loss=" + last);
            // Sauvegarde atomique après CHAQUE epoch : aucun progrès perdu.
            new java.io.File(CKPT_DIR).mkdirs();
            java.io.File tmp = new java.io.File(CKPT + ".tmp");
            model.saveState(tmp.getAbsolutePath());
            java.nio.file.Files.move(tmp.toPath(), ck.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            java.nio.file.Files.write(ce.toPath(), String.valueOf(done).getBytes());
        }
        System.out.println("RICH cumul=" + done + "/" + TARGET_EPOCHS + " (cette passe : " + first + " -> " + last + ")");

        double trainAcc = evaluate(model, new ArrayList<>(train.subList(0, 30)), "train(30)");
        double testAcc = evaluate(model, test, "unseen");
        System.out.println("RICH baseline=" + baseline + " train=" + trainAcc + " unseen=" + testAcc);

        org.junit.Assume.assumeTrue("Entraînement cumulatif en cours (" + done + "/" + TARGET_EPOCHS
                + " epochs) : relancer pour continuer ; assertions finales ignorées", done >= TARGET_EPOCHS);
        assertTrue("Le modèle doit battre le non entraîné (" + testAcc + " vs " + baseline + ")",
                testAcc > baseline + 0.3);
    }
}
