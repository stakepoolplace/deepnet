package RN.transformer;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.regex.Pattern;

import org.junit.Assume;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Prédiction du MOT SUIVANT (continuation de texte), entraînement cumulatif.
 *
 * Différence avec TransformerRichCorpusGeneralizationTest (recopie, conservé
 * tel quel avec son propre checkpoint) : ici l'entrée est un DÉBUT de phrase
 * et la cible est sa SUITE. Ainsi generate(amorce) prolonge réellement le texte.
 *
 * Évaluation objective sans phrase de référence unique : la phrase complétée
 * (amorce + suite générée) doit respecter la grammaire
 *   DET [ADJ] NOM VERBE [PREP] DET [ADJ] NOM
 * On mesure la fraction de complétions valides, sur des amorces tirées de
 * phrases JAMAIS vues, contre un modèle non entraîné.
 *
 * Checkpoint : target/transformer-nextword/ (jamais commité, indépendant de
 * celui du test de recopie). Chaque lancement reprend où le précédent s'est
 * arrêté ; les assertions ne s'appliquent qu'une fois TARGET_EPOCHS atteint.
 * Budget par lancement réglable : -Dnw.perRun=N.
 */
public class TransformerNextWordCumulativeTest {

    private static final int MAX_LEN = 12;
    private static final int D_MODEL = 64;
    private static final int TARGET_EPOCHS = Integer.getInteger("nw.target", 300);
    private static final int EPOCHS_PER_RUN = Integer.getInteger("nw.perRun", 20);
    private static final float LR = 0.001f;
    private static final int PREFIXES_PER_SENTENCE = 3;
    private static final String CKPT_DIR = System.getProperty("nw.dir", "target/transformer-nextword");
    private static final String CKPT = CKPT_DIR + "/model-v1-lr0001-seed42.ser";
    private static final String CKPT_EPOCHS = CKPT + ".epochs";
    // Meilleur checkpoint (selon la validité sur amorces inédites) : l'entraînement
    // peut diverger (pic de perte) et le dernier état n'est pas forcément le meilleur.
    private static final String BEST = CKPT_DIR + "/best.ser";
    private static final String BEST_SCORE = CKPT_DIR + "/best.score";

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

    private static String sentence(Random r) {
        StringBuilder sb = new StringBuilder();
        sb.append(pick(r, DET)).append(' ');
        if (r.nextBoolean()) sb.append(pick(r, ADJ)).append(' ');
        sb.append(pick(r, NOUN)).append(' ').append(pick(r, VERB)).append(' ');
        if (r.nextBoolean()) sb.append(pick(r, PREP)).append(' ');
        sb.append(pick(r, DET)).append(' ');
        if (r.nextBoolean()) sb.append(pick(r, ADJ)).append(' ');
        sb.append(pick(r, NOUN));
        return sb.toString();
    }

    // ---- validité grammaticale -------------------------------------------------
    private static char category(String w) {
        if (Arrays.asList(DET).contains(w)) return 'D';
        if (Arrays.asList(NOUN).contains(w)) return 'N';
        if (Arrays.asList(ADJ).contains(w)) return 'A';
        if (Arrays.asList(VERB).contains(w)) return 'V';
        if (Arrays.asList(PREP).contains(w)) return 'P';
        return '?';
    }

    private static final Pattern GRAMMAR = Pattern.compile("DA?NVP?DA?N");

    static boolean isValid(String sentence) {
        StringBuilder cats = new StringBuilder();
        for (String w : sentence.trim().split("\\s+")) {
            cats.append(category(w));
        }
        return GRAMMAR.matcher(cats).matches();
    }

    private static String norm(String s) {
        return s == null ? "" : s.replaceAll("<[A-Z]+>", " ").trim().replaceAll("\\s+", " ");
    }

    /** Fraction d'amorces dont la complétion (amorce + suite générée) est grammaticale. */
    private static double validity(TransformerModel m, List<String> prefixes, String label, boolean verbose) {
        int ok = 0;
        for (String p : prefixes) {
            String cont = norm(m.generate(p, 8, 0.05, 1, 7L));
            boolean valid = isValid(p + " " + cont);
            if (valid) ok++;
            if (verbose) {
                System.out.println("NEXTWORD " + label + (valid ? " OK  " : " KO  ") + "[" + p + "] + [" + cont + "]");
            }
        }
        System.out.println("NEXTWORD " + label + " complétions valides=" + ok + "/" + prefixes.size());
        return ok / (double) prefixes.size();
    }

    @Test
    public void testNextWordPredictionCumulative() throws Exception {
        Nd4j.getRandom().setSeed(42);
        Random rnd = new Random(2024);

        Set<String> all = new LinkedHashSet<>();
        while (all.size() < 400) all.add(sentence(rnd));
        List<String> pool = new ArrayList<>(all);
        List<String> testSentences = new ArrayList<>(pool.subList(0, 30));
        List<String> trainSentences = new ArrayList<>(pool.subList(30, 230)); // 200 phrases

        // Paires (début -> suite). Plusieurs débuts de longueurs variées par phrase.
        Random prng = new Random(99);
        List<String> in = new ArrayList<>(), out = new ArrayList<>();
        for (String s : trainSentences) {
            String[] w = s.split(" ");
            for (int j = 0; j < PREFIXES_PER_SENTENCE; j++) {
                int k = 1 + prng.nextInt(w.length - 1); // 1..n-1 mots en entrée
                in.add(String.join(" ", Arrays.copyOfRange(w, 0, k)));
                out.add(String.join(" ", Arrays.copyOfRange(w, k, w.length)));
            }
        }
        // Amorces de test : débuts de phrases JAMAIS vues (longueurs variées).
        List<String> testPrefixes = new ArrayList<>();
        for (String s : testSentences) {
            String[] w = s.split(" ");
            int k = 2 + prng.nextInt(w.length - 3);
            testPrefixes.add(String.join(" ", Arrays.copyOfRange(w, 0, k)));
        }
        for (String s : testSentences) {
            assertTrue("Phrase de test dans l'entraînement : " + s, !trainSentences.contains(s));
        }

        List<String> vocabulary = new ArrayList<>(Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>"));
        for (String[] a : new String[][] { DET, NOUN, ADJ, VERB, PREP }) vocabulary.addAll(Arrays.asList(a));
        System.out.println("NEXTWORD vocab=" + vocabulary.size() + " paires=" + in.size()
                + " amorces test=" + testPrefixes.size());

        Tokenizer tokenizer = new Tokenizer(vocabulary, D_MODEL, MAX_LEN);
        TransformerModel untrained = new TransformerModel(2, D_MODEL, 4, 256, 0.0, tokenizer.getVocabSize(),
                tokenizer, LR, 50);
        untrained.setTrained(true);
        double baseline = validity(untrained, testPrefixes, "baseline", false);

        Nd4j.getRandom().setSeed(42);
        TransformerModel model = new TransformerModel(2, D_MODEL, 4, 256, 0.0, tokenizer.getVocabSize(),
                tokenizer, LR, 50);
        model.getOptimizer().setMaxEpochs(Integer.MAX_VALUE);
        int done = 0;
        File ck = new File(CKPT), ce = new File(CKPT_EPOCHS);
        if (ck.exists() && ce.exists()) {
            model.loadState(CKPT);
            done = Integer.parseInt(new String(Files.readAllBytes(ce.toPath())).trim());
            System.out.println("NEXTWORD reprise du checkpoint : " + done + " epochs déjà faits");
        }

        DataGenerator gen = new DataGenerator(in, out, tokenizer, 16, MAX_LEN);
        float first = -1f, last = Float.NaN;
        int budget = Math.min(EPOCHS_PER_RUN, TARGET_EPOCHS - done);
        for (int e = 0; e < budget; e++) {
            last = model.train(gen, 1);
            assertTrue("Perte non finie : " + last, !Float.isNaN(last) && !Float.isInfinite(last));
            if (first < 0) first = last;
            done++;
            System.out.println("NEXTWORD epoch " + done + "/" + TARGET_EPOCHS + " loss=" + last);
            new File(CKPT_DIR).mkdirs();
            File tmp = new File(CKPT + ".tmp");
            model.saveState(tmp.getAbsolutePath());
            Files.move(tmp.toPath(), ck.toPath(), StandardCopyOption.REPLACE_EXISTING);
            Files.write(ce.toPath(), String.valueOf(done).getBytes());
        }
        System.out.println("NEXTWORD cumul=" + done + "/" + TARGET_EPOCHS + " (cette passe : " + first + " -> " + last + ")");

        double trainV = validity(model, new ArrayList<>(in.subList(0, 30)), "amorces-train", false);
        double testV = validity(model, testPrefixes, "amorces-inédites", true);
        if (budget > 0) {
            File bs = new File(BEST_SCORE);
            double prev = bs.exists() ? Double.parseDouble(new String(Files.readAllBytes(bs.toPath())).trim().split(" ")[0]) : -1;
            if (testV > prev) {
                Files.copy(ck.toPath(), new File(BEST).toPath(), StandardCopyOption.REPLACE_EXISTING);
                Files.write(bs.toPath(), (testV + " (epoch " + done + ")").getBytes());
                System.out.println("NEXTWORD nouveau MEILLEUR checkpoint : " + testV + " à l'epoch " + done);
            }
        }
        System.out.println("NEXTWORD baseline=" + baseline + " train=" + trainV + " inédites=" + testV);

        Assume.assumeTrue("Entraînement cumulatif en cours (" + done + "/" + TARGET_EPOCHS
                + " epochs) : relancer pour continuer ; assertions finales ignorées", done >= TARGET_EPOCHS);
        // Seuils provisoires, à recalibrer sur la première convergence complète.
        assertTrue("Complétions valides insuffisantes sur amorces inédites : " + testV + " (baseline " + baseline + ")",
                testV >= 0.5 && testV > baseline + 0.3);
    }
}
