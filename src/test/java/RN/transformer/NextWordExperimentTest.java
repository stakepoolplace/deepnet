package RN.transformer;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

import org.junit.Assume;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import RN.transformer.NextWordProtocol.EvalResult;
import RN.transformer.NextWordProtocol.Item;

/**
 * Expérience comparative reproductible (désactivée par défaut) :
 *   mvn test -Dtest=NextWordExperimentTest -Dnw.exp=diag   # diagnostic de best.ser sur VALIDATION seulement
 *   mvn test -Dtest=NextWordExperimentTest -Dnw.exp=full   # A vs B vs C + TEST (lu une seule fois)
 *
 * HISTORIQUE (honnêteté sur ce qui a été décidé avant/après mesure) :
 *  - Recette B fixée AVANT toute mesure sur les nouveaux jeux ;
 *  - le diagnostic (mode diag, VALIDATION uniquement) a montré que best.ser est SUR-APPRIS :
 *    perte d'entraînement 1,87 < oracle 1,94 (mémorisation) mais perte de validation 2,86
 *    (excès de 0,91 nat/token), validité 71,3 % sur 150 amorces. Le bras C a donc été AJOUTÉ
 *    après ce diagnostic (il n'est pas pré-enregistré) ; B est conservé tel que pré-enregistré.
 *  - le jeu TEST n'a été lu par aucune mesure avant la fin de ce test.
 *
 * ARMS
 *  A : best.ser tel quel (référence).
 *  B : reprise de best.ser. Moments d'Adam et pas restaurés ; LR cosinus pic 2e-4, warmup 20 pas,
 *      plancher 5 %, sur BUDGET_B epochs ; clipping norme globale (seuil = 2 x médiane des normes
 *      d'une epoch d'un modèle jetable à LR nul) ; dropout 0 ; early stopping.
 *  C : depuis zéro, même corpus, dropout 0,1, LR cosinus pic 1e-3, warmup 100 pas, plancher 5 %
 *      sur BUDGET_C epochs, même clipping, early stopping ; (modèle plus régularisé + sélection fiable).
 *  Early stopping commun : perte de VALIDATION (150 amorces inédites) toutes les 5 epochs, patience
 *  4 évaluations (3 pour B), minDelta 0,002 nat ; le modèle retenu est le meilleur point selon la
 *  validation, jamais le dernier. La validité mesurée en cours de route est INFORMATIVE seulement.
 *
 * RÈGLE DE DÉCISION (deux comparaisons, Bonferroni : seuil 0,025) : un bras n'est « meilleur que A »
 * que si, sur TEST, le test des signes apparié sur la validité donne p < 0,025 en sa faveur.
 * L'objectif 90 % n'est « atteint » que si la borne basse de l'IC95 de Wilson sur TEST est >= 90 %
 * (sinon : estimation seulement, ou non atteint).
 */
public class NextWordExperimentTest {

    private static final String START_CKPT = "target/transformer-nextword/best.ser";
    private static final String DIR = "target/nextword-experiment";
    private static final int BUDGET_B = 60;
    private static final int BUDGET_C = 80;
    private static final int EVAL_EVERY = 5;
    private static final int STEPS_PER_EPOCH = 38; // ceil(600 / 16)
    private static final float PEAK_LR = 2e-4f;

    private static Tokenizer tok;

    private static TransformerModel build() {
        return build(0.0);
    }

    private static TransformerModel build(double dropout) {
        Nd4j.getRandom().setSeed(42);
        tok = new Tokenizer(NextWordProtocol.vocabulary(), 64, NextWordProtocol.MAX_LEN);
        TransformerModel m = new TransformerModel(2, 64, 4, 256, dropout, tok.getVocabSize(), tok, 0.001f, 50);
        m.getOptimizer().setMaxEpochs(Integer.MAX_VALUE);
        return m;
    }

    private static DataGenerator gen(List<Item> items) {
        List<String> in = new ArrayList<>(), out = new ArrayList<>();
        for (Item it : items) { in.add(it.prefix); out.add(it.continuation); }
        return new DataGenerator(in, out, tok, 16, NextWordProtocol.MAX_LEN);
    }

    private static void log(StringBuilder rep, String line) {
        System.out.println("EXP " + line);
        rep.append(line).append('\n');
    }

    @Test
    public void experiment() throws Exception {
        String mode = System.getProperty("nw.exp");
        Assume.assumeTrue("Expérience désactivée (-Dnw.exp=diag|full)", mode != null);
        new File(DIR).mkdirs();
        StringBuilder rep = new StringBuilder();

        File start = new File(DIR + "/start-copy-of-best.ser");
        Files.copy(new File(START_CKPT).toPath(), start.toPath(), StandardCopyOption.REPLACE_EXISTING);
        String bestScore = new String(Files.readAllBytes(new File("target/transformer-nextword/best.score").toPath()));
        log(rep, "départ best.ser (" + bestScore.trim() + ")");
        log(rep, "plancher théorique de la perte (population) = " + NextWordProtocol.populationLossFloor());

        List<Item> val = NextWordProtocol.validationItems();
        List<Item> test = NextWordProtocol.testItems();
        List<Item> trainSample = NextWordProtocol.trainItems().subList(0, 150);

        TransformerModel a = build();
        a.loadState(start.getAbsolutePath());
        EvalResult aVal = NextWordProtocol.evaluate(a, tok, val, "A validation");
        log(rep, aVal.toString());
        log(rep, NextWordProtocol.evaluate(a, tok, trainSample, "A train(150)").toString());
        // Diagnostic des échecs (validation) : quels types d'erreurs et à quelle position
        int shown = 0;
        for (int i = 0; i < val.size() && shown < 12; i++) {
            if (!aVal.validFlags[i]) {
                log(rep, "  échec A: [" + val.get(i).prefix + "] + [" + aVal.outputs[i] + "]  (attendu: ["
                        + val.get(i).continuation + "])");
                shown++;
            }
        }
        if ("diag".equals(mode)) {
            Files.write(new File(DIR + "/report-diag.txt").toPath(), rep.toString().getBytes());
            return;
        }

        // ---- seuil de clipping : médiane des normes sur UNE epoch d'un modèle jetable à LR nul
        TransformerModel probe = build();
        probe.loadState(start.getAbsolutePath());
        probe.getOptimizer().configureSchedule(CustomAdamOptimizer.Schedule.CONSTANT, 0f, 0, 1, 0f);
        List<Double> norms = new ArrayList<>();
        probe.setGradNormRecorder(norms::add);
        probe.train(gen(NextWordProtocol.trainItems()), 1);
        java.util.Collections.sort(norms);
        double median = norms.get(norms.size() / 2);
        double clip = 2 * median;
        log(rep, String.format("normes de gradient (1 epoch, %d pas) : min=%.2f médiane=%.2f max=%.2f -> seuil clipping=%.2f",
                norms.size(), norms.get(0), median, norms.get(norms.size() - 1), clip));

        int[] clipped = new int[1], total = new int[1];

        // ---- arm B : reprise de best.ser (dropout 0, architecture identique)
        TransformerModel b = build(0.0);
        b.loadState(start.getAbsolutePath());
        log(rep, "B : état optimiseur restauré (pas=" + b.getOptimizer().getCurrentStep() + ", moments Adam inclus)");
        File bBest = new File(DIR + "/B-best-by-validation.ser");
        runArm(rep, "B", b, bBest, CustomAdamOptimizer.Schedule.COSINE, 2e-4f, 20, BUDGET_B, 3, clip, val, clipped, total);

        // ---- arm C : depuis zéro, dropout 0.1
        TransformerModel c = build(0.1);
        File cBest = new File(DIR + "/C-best-by-validation.ser");
        runArm(rep, "C", c, cBest, CustomAdamOptimizer.Schedule.COSINE, 1e-3f, 100, BUDGET_C, 4, clip, val, clipped, total);

        TransformerModel bSel = build(0.0);
        bSel.loadState(bBest.getAbsolutePath());
        TransformerModel cSel = build(0.1);
        cSel.loadState(cBest.getAbsolutePath());
        cSel.setTrained(true);

        // ---- comparaison, même protocole exactement (dropout désactivé à l'inférence par construction)
        List<EvalResult> valRes = new ArrayList<>();
        valRes.add(aVal);
        valRes.add(NextWordProtocol.evaluate(bSel, tok, val, "B validation"));
        valRes.add(NextWordProtocol.evaluate(cSel, tok, val, "C validation"));
        EvalResult aTest = NextWordProtocol.evaluate(a, tok, test, "A TEST");
        EvalResult bTest = NextWordProtocol.evaluate(bSel, tok, test, "B TEST");
        EvalResult cTest = NextWordProtocol.evaluate(cSel, tok, test, "C TEST");
        log(rep, "===== RÉSULTATS (protocole identique) =====");
        for (EvalResult r : valRes) log(rep, r.toString());
        log(rep, aTest.toString());
        log(rep, bTest.toString());
        log(rep, cTest.toString());
        for (EvalResult arm : new EvalResult[] { bTest, cTest }) {
            int better = 0, worse = 0;
            for (int i = 0; i < test.size(); i++) {
                if (arm.validFlags[i] && !aTest.validFlags[i]) better++;
                if (!arm.validFlags[i] && aTest.validFlags[i]) worse++;
            }
            double p = NextWordProtocol.signTestP(better, worse);
            double[] ci = NextWordProtocol.wilson(arm.valid, arm.n);
            log(rep, String.format("%s vs A (TEST apparié) : meilleur sur %d, pire sur %d, p(signes)=%.4f -> %s",
                    arm.label, better, worse, p, (better > worse && p < 0.025) ? "AMÉLIORATION significative"
                            : "amélioration NON démontrée"));
            log(rep, String.format("%s objectif 90%% : estimation %.1f%%, borne basse IC95 %.1f%% -> %s", arm.label,
                    100 * arm.validity(), 100 * ci[0],
                    ci[0] >= 0.9 ? "ATTEINT" : (arm.validity() >= 0.9 ? "estimation seulement, non prouvé" : "NON atteint")));
        }
        Files.write(new File(DIR + "/report-full.txt").toPath(), rep.toString().getBytes());
    }

    /** Entraîne un bras avec planning, clipping global et early stopping sur la perte de validation. */
    private static void runArm(StringBuilder rep, String name, TransformerModel m, File bestFile,
            CustomAdamOptimizer.Schedule sched, float peak, int warmup, int budget, int patience,
            double clip, List<Item> val, int[] clipped, int[] total) throws Exception {
        m.setGradientClipping(TransformerModel.ClipMode.GLOBAL_NORM, clip);
        clipped[0] = 0;
        total[0] = 0;
        m.setGradNormRecorder(n -> { total[0]++; if (n > clip) clipped[0]++; });
        m.getOptimizer().configureSchedule(sched, peak, warmup, budget * STEPS_PER_EPOCH, 0.05f);
        EarlyStopping es = new EarlyStopping(patience, 0.002);
        DataGenerator trainGen = gen(NextWordProtocol.trainItems());
        DataGenerator valGen = gen(val);
        m.setTrained(true);
        float v0 = m.evaluateLoss(valGen);
        es.update(v0, 0);
        m.saveState(bestFile.getAbsolutePath());
        log(rep, name + " epoch 0 : val NLL=" + v0);
        for (int e = 1; e <= budget; e++) {
            float tl = m.train(trainGen, 1);
            if (e % EVAL_EVERY == 0) {
                float vl = m.evaluateLoss(valGen);
                boolean improved = es.update(vl, e);
                log(rep, String.format("%s epoch %d : train=%.4f valNLL=%.4f lr=%.2e clipping actif=%d/%d pas%s", name, e,
                        tl, vl, m.getOptimizer().getLearningRate(), clipped[0], total[0], improved ? "  <- meilleur" : ""));
                if (improved) m.saveState(bestFile.getAbsolutePath());
                if (es.shouldStop()) {
                    log(rep, name + " early stopping à l'epoch " + e);
                    break;
                }
            }
        }
        log(rep, name + " meilleur point selon la validation : epoch " + es.getBestStep() + " (val NLL " + es.getBest() + ")");
    }
}
