package RN.transformer;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.junit.Assume;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import RN.transformer.NextWordProtocol.EvalResult;
import RN.transformer.NextWordProtocol.Item;

/**
 * Expériences contrôlées v2 (désactivées par défaut ; -Dnw2.mode=train|final).
 *
 * MODE train : entraîne UNE recette depuis zéro (graine 42, architecture inchangée : 2 couches, dModel 64,
 *   4 têtes, dff 256, dropout 0) et écrit dans target/nextword-experiment2/<tag>/ :
 *   curve.csv (par epoch), best.ser (meilleure perte de VALIDATION, évaluée toutes les 5 epochs),
 *   last.ser, summary.txt. Propriétés (défauts = recette numérique corrigée) :
 *     nw2.tag, nw2.epochs=100, nw2.data=fixed|dynamic, nw2.norm=MEAN|SUM, nw2.clip=GLOBAL|PER_TENSOR,
 *     nw2.clipNorm=1.0, nw2.sched=COSINE|LEGACY, nw2.lr=5e-4, nw2.warmup=100, nw2.minRatio=0.05
 *   La validité (grammaticale) n'est que MONITORÉE (toutes les 10 epochs) ; la sélection se fait sur la
 *   perte de validation. Aucune lecture de TEST pendant l'entraînement.
 *
 * MODE final : charge les checkpoints listés (nw2.arms="nom=chemin,nom=chemin"), les évalue sur VALIDATION,
 *   désigne le gagnant sur VALIDATION SEULE (écrit selection.txt AVANT toute lecture de TEST-FINAL), puis
 *   évalue tous les bras sur TEST-FINAL (jamais lu avant) avec IC de Wilson et tests des signes appariés.
 *   L'objectif 90 % n'est « atteint » que si la borne basse IC95 du GAGNANT SÉLECTIONNÉ SUR VALIDATION >= 90 %.
 */
public class NextWordExperiment2Test {

    private static final String DIR = "target/nextword-experiment2";
    private static Tokenizer tok;

    private static TransformerModel build() {
        Nd4j.getRandom().setSeed(42);
        tok = new Tokenizer(vocabC(), 64, maxLenC());
        TransformerModel m = new TransformerModel(2, 64, 4, 256, 0.0, tok.getVocabSize(), tok, 0.001f, 50);
        m.getOptimizer().setMaxEpochs(Integer.MAX_VALUE);
        return m;
    }

    private static DataGenerator gen(List<Item> items) {
        List<String> in = new ArrayList<>(), out = new ArrayList<>();
        for (Item it : items) { in.add(it.prefix); out.add(it.continuation); }
        return new DataGenerator(in, out, tok, 16, maxLenC());
    }

    private static String prop(String k, String d) { return System.getProperty(k, d); }

    // ---- interrupteur de corpus : -Dnw2.corpus=v1 (défaut, inchangé) | v2 (accords + sémantique)
    private static boolean v2() { return "v2".equals(System.getProperty("nw2.corpus", "v1")); }
    private static List<String> vocabC() { return v2() ? CorpusV2.vocabulary() : NextWordProtocol.vocabulary(); }
    private static int maxLenC() { return v2() ? CorpusV2.MAX_LEN : NextWordProtocol.MAX_LEN; }
    private static List<Item> valItemsC() { return v2() ? CorpusV2.validationItems() : NextWordProtocol.validationItems(); }
    private static List<Item> trainItemsC() { return v2() ? CorpusV2.trainItems() : NextWordProtocol.trainItems(); }
    private static List<Item> testFinalItemsC() { return v2() ? CorpusV2.testFinalItems() : NextWordProtocol.testFinalItems(); }
    private static Set<String> forbiddenC() { return v2() ? CorpusV2.forbiddenForTraining() : NextWordProtocol.forbiddenForTraining(); }
    private static NextWordProtocol.Sample dynamicC(long seed, int epoch, Set<String> forb) {
        return v2() ? CorpusV2.dynamicSample(seed, epoch, 200, 3, forb) : NextWordProtocol.dynamicSample(seed, epoch, 200, 3, forb);
    }

    private static double median(List<Double> l) {
        List<Double> c = new ArrayList<>(l);
        Collections.sort(c);
        return c.get(c.size() / 2);
    }

    @Test
    public void experiment() throws Exception {
        String mode = System.getProperty("nw2.mode");
        Assume.assumeTrue("désactivé (-Dnw2.mode=train|final)", mode != null);
        if ("train".equals(mode)) train(); else if (v2()) finalReportV2(); else finalReport();
    }

    // ================================================================== train
    private void train() throws Exception {
        String tag = prop("nw2.tag", "run");
        int epochs = Integer.parseInt(prop("nw2.epochs", "100"));
        boolean dynamic = "dynamic".equals(prop("nw2.data", "fixed"));
        TransformerModel.LossNormalization norm = TransformerModel.LossNormalization.valueOf(
                "SUM".equals(prop("nw2.norm", "MEAN")) ? "SUM" : "MEAN_PER_TOKEN");
        boolean globalClip = "GLOBAL".equals(prop("nw2.clip", "GLOBAL"));
        double clipNorm = Double.parseDouble(prop("nw2.clipNorm", "1.0"));
        boolean legacySched = "LEGACY".equals(prop("nw2.sched", "COSINE"));
        float lr = Float.parseFloat(prop("nw2.lr", "5e-4"));
        int warmup = Integer.parseInt(prop("nw2.warmup", "100"));
        float minRatio = Float.parseFloat(prop("nw2.minRatio", "0.05"));
        int stepsPerEpoch = 38;
        int horizon = Integer.parseInt(prop("nw2.horizon", String.valueOf(epochs)));

        File dir = new File(DIR + "/" + tag);
        dir.mkdirs();
        TransformerModel m = build();
        m.setLossNormalization(norm);
        m.setGradientClipping(globalClip ? TransformerModel.ClipMode.GLOBAL_NORM : TransformerModel.ClipMode.PER_TENSOR,
                globalClip ? clipNorm : 0.5);
        if (!legacySched) {
            m.getOptimizer().configureSchedule(CustomAdamOptimizer.Schedule.COSINE, lr, warmup, horizon * stepsPerEpoch, minRatio);
        }
        m.setTrained(true);

        List<Double> batchLosses = new ArrayList<>(), gradNorms = new ArrayList<>();
        List<Double> epochMaxRatio = new ArrayList<>();
        double[] curMaxRatio = { 0 };
        String[] curArgShape = { "" };
        List<Double> ratios = new ArrayList<>();
        m.setBatchLossRecorder(batchLosses::add);
        m.setGradNormRecorder(gradNorms::add);
        m.getOptimizer().setDiagnostics(d -> {
            double r = d.maxUpdateRatio();
            ratios.add(r);
            if (r > curMaxRatio[0]) {
                curMaxRatio[0] = r;
                curArgShape[0] = Arrays.toString(d.shapes[d.argMaxUpdateRatio()]);
            }
        });

        List<Item> val = valItemsC();
        DataGenerator valGen = gen(val);
        DataGenerator fixedGen = gen(trainItemsC());
        int fixedTokens = 0;
        for (Item it : trainItemsC()) fixedTokens += v2() ? CorpusV2.tokenCount(it) : NextWordProtocol.tokenCount(it);
        Set<String> forbidden = forbiddenC();

        EarlyStopping best = new EarlyStopping(Integer.MAX_VALUE, 0.0);
        long totalTokens = 0;
        int totalSpikes = 0;
        try (PrintWriter csv = new PrintWriter(new FileWriter(new File(dir, "curve.csv")))) {
            csv.println("epoch,trainLoss,valNLL,valValidity,supervisedTokens,lr,gradNormMedian,gradNormMax,"
                    + "maxUpdateRatio,maxUpdateRatioTensor,spikes,maxBatchOverMedian");
            double v0 = m.evaluateLoss(valGen);
            System.out.println("EXP2 " + tag + " epoch 0 valNLL=" + v0);
            for (int e = 1; e <= epochs; e++) {
                int tokens = fixedTokens;
                DataGenerator g = fixedGen;
                if (dynamic) {
                    NextWordProtocol.Sample sm = dynamicC(42, e, forbidden);
                    g = gen(sm.items);
                    tokens = sm.supervisedTokens;
                }
                totalTokens += tokens;
                int lossStart = batchLosses.size(), gnStart = gradNorms.size();
                curMaxRatio[0] = 0;
                float tl = m.train(g, 1);
                // pics : perte de batch > 1.5 x médiane glissante des 76 précédentes
                int spikes = 0;
                double maxOver = 0;
                for (int i = lossStart; i < batchLosses.size(); i++) {
                    int from = Math.max(0, i - 76);
                    if (i - from >= 38) {
                        double med = median(batchLosses.subList(from, i));
                        maxOver = Math.max(maxOver, batchLosses.get(i) / med);
                        if (batchLosses.get(i) > 1.5 * med) spikes++;
                    }
                }
                totalSpikes += spikes;
                List<Double> gn = new ArrayList<>(gradNorms.subList(gnStart, gradNorms.size()));
                String valNll = "", valValid = "";
                if (e % 5 == 0) {
                    float vl = m.evaluateLoss(valGen);
                    valNll = String.valueOf(vl);
                    if (best.update(vl, e)) {
                        m.saveState(new File(dir, "best.ser").getAbsolutePath());
                    }
                    if (e % 10 == 0) {
                        if (v2()) {
                            CorpusV2.EvalV2 r2 = CorpusV2.evaluate(m, tok, val, "val");
                            valValid = String.valueOf(r2.l3 / (double) r2.n);
                            try (PrintWriter lv = new PrintWriter(new FileWriter(new File(dir, "levels.csv"), true))) {
                                lv.println(e + "," + r2.l1 + "," + r2.l2 + "," + r2.l3 + "," + r2.n + "," + r2.firstViolations);
                            }
                        } else {
                            valValid = String.valueOf(NextWordProtocol.evaluate(m, tok, val, "val").validity());
                        }
                    }
                }
                String line = String.format(java.util.Locale.ROOT, "%d,%.5f,%s,%s,%d,%.3e,%.4f,%.4f,%.4f,\"%s\",%d,%.3f",
                        e, tl, valNll, valValid, tokens, m.getOptimizer().getLearningRate(), median(gn),
                        Collections.max(gn), curMaxRatio[0], curArgShape[0], spikes, maxOver);
                csv.println(line);
                csv.flush();
                System.out.println("EXP2 " + tag + " " + line);
            }
        }
        m.saveState(new File(dir, "last.ser").getAbsolutePath());
        String summary = String.format(java.util.Locale.ROOT,
                "tag=%s epochs=%d data=%s norm=%s clip=%s(%s) sched=%s lr=%s tokens_supervises_total=%d "
                        + "meilleure_valNLL=%.4f (epoch %d) pics_total=%d ratioMaj_mediane=%.5f ratioMaj_max=%.4f",
                tag, epochs, dynamic ? "dynamic" : "fixed", norm, globalClip ? "GLOBAL" : "PER_TENSOR", clipNorm,
                legacySched ? "LEGACY" : "COSINE", lr, totalTokens, best.getBest(), best.getBestStep(), totalSpikes,
                median(ratios), Collections.max(ratios));
        Files.write(new File(dir, "summary.txt").toPath(), summary.getBytes());
        System.out.println("EXP2 SUMMARY " + summary);
    }

    // ================================================================== final
    private void finalReport() throws Exception {
        String[] arms = prop("nw2.arms", "").split(",");
        Assume.assumeTrue("nw2.arms requis", arms.length > 0 && !arms[0].isEmpty());
        new File(DIR).mkdirs();
        StringBuilder rep = new StringBuilder();
        List<String> names = new ArrayList<>();
        List<TransformerModel> models = new ArrayList<>();
        for (String a : arms) {
            String[] kv = a.split("=");
            File copy = new File(DIR + "/historical-" + kv[0] + ".ser");
            Files.copy(new File(kv[1]).toPath(), copy.toPath(), StandardCopyOption.REPLACE_EXISTING);
            TransformerModel m = build();
            m.loadState(copy.getAbsolutePath());
            names.add(kv[0]);
            models.add(m);
        }
        List<Item> val = NextWordProtocol.validationItems();
        log(rep, "plancher théorique de la perte = " + NextWordProtocol.populationLossFloor());
        List<EvalResult> valRes = new ArrayList<>();
        int winner = 0;
        for (int i = 0; i < models.size(); i++) {
            EvalResult r = NextWordProtocol.evaluate(models.get(i), tok, val, names.get(i) + " VALIDATION");
            valRes.add(r);
            log(rep, r.toString());
            EvalResult w = valRes.get(winner);
            if (r.nll < w.nll - 1e-9) winner = i;
        }
        String sel = "GAGNANT sélectionné sur VALIDATION seule (perte de validation minimale) : " + names.get(winner);
        log(rep, sel);
        Files.write(new File(DIR + "/selection.txt").toPath(), sel.getBytes()); // écrit AVANT TEST-FINAL

        List<Item> tf = NextWordProtocol.testFinalItems();
        List<EvalResult> tfRes = new ArrayList<>();
        for (int i = 0; i < models.size(); i++) {
            EvalResult r = NextWordProtocol.evaluate(models.get(i), tok, tf, names.get(i) + " TEST-FINAL");
            tfRes.add(r);
            log(rep, r.toString());
        }
        EvalResult ref = tfRes.get(0);
        for (int i = 1; i < tfRes.size(); i++) {
            int better = 0, worse = 0;
            for (int k = 0; k < tf.size(); k++) {
                if (tfRes.get(i).validFlags[k] && !ref.validFlags[k]) better++;
                if (!tfRes.get(i).validFlags[k] && ref.validFlags[k]) worse++;
            }
            log(rep, String.format("%s vs %s (TEST-FINAL apparié) : meilleur sur %d, pire sur %d, p(signes)=%.4f",
                    names.get(i), names.get(0), better, worse, NextWordProtocol.signTestP(better, worse)));
        }
        EvalResult w = tfRes.get(winner);
        double[] ci = NextWordProtocol.wilson(w.valid, w.n);
        log(rep, String.format("VERDICT objectif 90%% (gagnant %s, sélectionné sur validation) : estimation %.1f%%, "
                + "IC95 %.1f–%.1f%% -> %s", names.get(winner), 100 * w.validity(), 100 * ci[0], 100 * ci[1],
                ci[0] >= 0.9 ? "ATTEINT (borne basse >= 90%)" : (w.validity() >= 0.9
                        ? "estimation >= 90% mais NON prouvé (borne basse < 90%)" : "NON atteint")));
        Files.write(new File(DIR + "/report-final.txt").toPath(), rep.toString().getBytes());
    }

    // ================================================================== final (corpus v2)
    private void finalReportV2() throws Exception {
        String[] arms = prop("nw2.arms", "").split(",");
        Assume.assumeTrue("nw2.arms requis", arms.length > 0 && !arms[0].isEmpty());
        String out = DIR + "/v2-final";
        new File(out).mkdirs();
        StringBuilder rep = new StringBuilder();
        List<String> names = new ArrayList<>();
        List<TransformerModel> models = new ArrayList<>();
        for (String a : arms) {
            String[] kv = a.split("=");
            TransformerModel m = build();
            if (!"INIT".equals(kv[1])) {
                File copy = new File(out + "/historical-" + kv[0] + ".ser");
                Files.copy(new File(kv[1]).toPath(), copy.toPath(), StandardCopyOption.REPLACE_EXISTING);
                m.loadState(copy.getAbsolutePath());
            } else {
                m.setTrained(true); // référence : modèle non entraîné
            }
            names.add(kv[0]);
            models.add(m);
        }
        double[] floor = CorpusV2.populationLossFloor(20000, 123);
        log(rep, String.format("CORPUS V2 : plancher théorique de la perte = %.4f ± %.4f", floor[0], floor[1]));
        List<Item> val = CorpusV2.validationItems();
        List<CorpusV2.EvalV2> valRes = new ArrayList<>();
        int winner = -1;
        for (int i = 0; i < models.size(); i++) {
            CorpusV2.EvalV2 r = CorpusV2.evaluate(models.get(i), tok, val, names.get(i) + " VALIDATION");
            valRes.add(r);
            log(rep, r.toString());
            if (!"INIT".equals(arms[i].split("=")[1]) && (winner < 0 || r.nll < valRes.get(winner).nll - 1e-9)) winner = i;
        }
        String sel = "GAGNANT sélectionné sur VALIDATION seule (perte de validation minimale, hors référence INIT) : " + names.get(winner);
        log(rep, sel);
        Files.write(new File(out + "/selection.txt").toPath(), sel.getBytes()); // avant toute lecture de TEST-FINAL

        List<Item> tf = CorpusV2.testFinalItems();
        List<CorpusV2.EvalV2> tfRes = new ArrayList<>();
        for (int i = 0; i < models.size(); i++) {
            CorpusV2.EvalV2 r = CorpusV2.evaluate(models.get(i), tok, tf, names.get(i) + " TEST-FINAL");
            tfRes.add(r);
            log(rep, r.toString());
        }
        for (int i = 0; i < tfRes.size(); i++) {
            for (int j = i + 1; j < tfRes.size(); j++) {
                int better = 0, worse = 0;
                for (int k = 0; k < tf.size(); k++) {
                    if (tfRes.get(j).l3Flags[k] && !tfRes.get(i).l3Flags[k]) better++;
                    if (!tfRes.get(j).l3Flags[k] && tfRes.get(i).l3Flags[k]) worse++;
                }
                log(rep, String.format("%s vs %s (TEST-FINAL apparié, N3) : meilleur sur %d, pire sur %d, p(signes)=%.4f",
                        names.get(j), names.get(i), better, worse, NextWordProtocol.signTestP(better, worse)));
            }
        }
        CorpusV2.EvalV2 w = tfRes.get(winner);
        double[] ci = NextWordProtocol.wilson(w.l3, w.n);
        log(rep, String.format("VERDICT objectif 90%% au niveau N3 (gagnant %s, sélectionné sur validation) : %.1f%%, IC95 %.1f–%.1f%% -> %s",
                names.get(winner), 100.0 * w.l3 / w.n, 100 * ci[0], 100 * ci[1],
                ci[0] >= 0.9 ? "ATTEINT (borne basse >= 90%)" : (w.l3 / (double) w.n >= 0.9
                        ? "estimation >= 90% mais NON prouvé" : "NON atteint")));
        Files.write(new File(out + "/report-final.txt").toPath(), rep.toString().getBytes());
    }

    private static void log(StringBuilder rep, String line) {
        System.out.println("EXP2 " + line);
        rep.append(line).append('\n');
    }
}
