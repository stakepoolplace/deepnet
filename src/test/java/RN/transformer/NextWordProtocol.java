package RN.transformer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Protocole FIXE et déterministe pour la prédiction du mot suivant sur le corpus
 * généré par grammaire (corpus initial conservé : mêmes 200 phrases / 600 paires
 * d'entraînement que TransformerNextWordCumulativeTest).
 *
 * Jeux (tous issus du même flux Random(2024), sans aucun recouvrement de PHRASES) :
 *  - SELECTION_OLD  pool[0,30)    : l'ancien jeu de 30 amorces, utilisé pour choisir
 *                                   best.ser -> CONTAMINÉ pour la sélection, exclu des
 *                                   jeux de validation/test ;
 *  - TRAIN          pool[30,230)  : 200 phrases, 3 amorces chacune (Random(99)) ;
 *  - VALIDATION     pool[230,380) : 150 phrases, 1 amorce chacune (Random(31337)) ;
 *  - TEST           pool[380,580) : 200 phrases, 1 amorce chacune (Random(4242)).
 * Les longueurs d'amorce suivent la même loi que l'entraînement (uniforme 1..n-1).
 * VALIDATION sert à l'early stopping / la sélection ; TEST n'est lu qu'à la fin.
 *
 * Trois niveaux de métrique, volontairement distincts :
 *  1. validité grammaticale : amorce + suite respecte DET [ADJ] NOM VERBE [PREP] DET [ADJ] NOM ;
 *  2. correspondance exacte avec la suite attendue (plafond théorique fourni par l'oracle) ;
 *  3. validité SÉMANTIQUE : non définie sur ce corpus (aucune contrainte de sens, accords
 *     compris : « une enfant », « la oiseau » sont des phrases VALIDES du corpus).
 *     Ne pas la mesurer est plus honnête que d'en inventer une.
 */
public class NextWordProtocol {

    public static final int MAX_LEN = 12;
    public static final String[] DET = { "le", "la", "les", "un", "une" };
    public static final String[] NOUN = { "chat", "chien", "oiseau", "enfant", "livre", "table", "jardin",
            "maison", "voiture", "arbre", "riviere", "montagne", "ville", "route", "cheval" };
    public static final String[] ADJ = { "petit", "grand", "rouge", "vert", "vieux", "joli", "calme", "rapide" };
    public static final String[] VERB = { "voit", "aime", "cherche", "trouve", "garde", "suit", "regarde",
            "dessine" };
    public static final String[] PREP = { "dans", "sur", "pres", "devant", "derriere", "avec" };

    /** Un exemple : amorce, suite attendue, phrase complète. */
    public static class Item {
        public final String prefix;
        public final String continuation;

        Item(String prefix, String continuation) {
            this.prefix = prefix;
            this.continuation = continuation;
        }

        public String full() {
            return prefix + " " + continuation;
        }
    }

    // ------------------------------------------------------------------ corpus
    private static String pick(Random r, String[] a) {
        return a[r.nextInt(a.length)];
    }

    static String sentence(Random r) {
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

    public static List<String> pool() {
        Random r = new Random(2024);
        Set<String> all = new LinkedHashSet<>();
        while (all.size() < 1000) all.add(sentence(r));
        return new ArrayList<>(all);
    }

    public static List<String> trainSentences() { return new ArrayList<>(pool().subList(30, 230)); }
    public static List<String> selectionOldSentences() { return new ArrayList<>(pool().subList(0, 30)); }
    public static List<String> validationSentences() { return new ArrayList<>(pool().subList(230, 380)); }
    public static List<String> testSentences() { return new ArrayList<>(pool().subList(380, 580)); }

    /**
     * TEST-FINAL : jeu neuf, jamais lu par aucune mesure avant le verdict final (le jeu TEST précédent a
     * été lu une fois pour comparer A/B/C ; il reste intact mais n'est plus « vierge »).
     */
    public static List<String> testFinalSentences() { return new ArrayList<>(pool().subList(600, 800)); }
    public static List<Item> testFinalItems() { return oneItemPerSentence(testFinalSentences(), 777L); }

    /** Toutes les phrases interdites à l'échantillonnage dynamique : pool[0, 800) (validation, tests, ancien corpus). */
    public static Set<String> forbiddenForTraining() { return new HashSet<>(pool().subList(0, 800)); }

    /** Résultat d'un échantillonnage dynamique : paires + nombre de tokens supervisés (mots de la suite + END). */
    public static class Sample {
        public final List<Item> items = new ArrayList<>();
        public int supervisedTokens;
    }

    /**
     * Échantillonne `sentences` phrases NOUVELLES depuis la grammaire (graine reproductible dérivée de
     * (seed, epoch)), sans doublon dans l'epoch et hors `forbidden`, puis `prefixesPerSentence` amorces
     * par phrase (k uniforme 1..n-1, comme le corpus fixe). Même nombre de paires par epoch que le
     * corpus fixe (200 x 3 = 600) : budget en exemples comparable, budget en tokens mesuré.
     */
    public static Sample dynamicSample(long seed, int epoch, int sentences, int prefixesPerSentence,
            Set<String> forbidden) {
        Random r = new Random(seed * 1_000_003L + epoch);
        Set<String> chosen = new LinkedHashSet<>();
        while (chosen.size() < sentences) {
            String s = sentence(r);
            if (!forbidden.contains(s)) chosen.add(s); // le LinkedHashSet écarte les doublons
        }
        Sample out = new Sample();
        for (String s : chosen) {
            String[] w = s.split(" ");
            for (int j = 0; j < prefixesPerSentence; j++) {
                int k = 1 + r.nextInt(w.length - 1);
                Item it = new Item(String.join(" ", Arrays.copyOfRange(w, 0, k)),
                        String.join(" ", Arrays.copyOfRange(w, k, w.length)));
                out.items.add(it);
                out.supervisedTokens += tokenCount(it);
            }
        }
        return out;
    }

    /** 600 paires d'entraînement : IDENTIQUES à celles de TransformerNextWordCumulativeTest. */
    public static List<Item> trainItems() {
        Random prng = new Random(99);
        List<Item> items = new ArrayList<>();
        for (String s : trainSentences()) {
            String[] w = s.split(" ");
            for (int j = 0; j < 3; j++) {
                int k = 1 + prng.nextInt(w.length - 1);
                items.add(new Item(String.join(" ", Arrays.copyOfRange(w, 0, k)),
                        String.join(" ", Arrays.copyOfRange(w, k, w.length))));
            }
        }
        return items;
    }

    private static List<Item> oneItemPerSentence(List<String> sentences, long seed) {
        Random prng = new Random(seed);
        List<Item> items = new ArrayList<>();
        for (String s : sentences) {
            String[] w = s.split(" ");
            int k = 1 + prng.nextInt(w.length - 1);
            items.add(new Item(String.join(" ", Arrays.copyOfRange(w, 0, k)),
                    String.join(" ", Arrays.copyOfRange(w, k, w.length))));
        }
        return items;
    }

    public static List<Item> validationItems() { return oneItemPerSentence(validationSentences(), 31337L); }
    public static List<Item> testItems() { return oneItemPerSentence(testSentences(), 4242L); }

    public static List<String> vocabulary() {
        List<String> v = new ArrayList<>(Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>"));
        for (String[] a : new String[][] { DET, NOUN, ADJ, VERB, PREP }) v.addAll(Arrays.asList(a));
        return v;
    }

    // ------------------------------------------------------------- grammaire
    static char category(String w) {
        if (Arrays.asList(DET).contains(w)) return 'D';
        if (Arrays.asList(NOUN).contains(w)) return 'N';
        if (Arrays.asList(ADJ).contains(w)) return 'A';
        if (Arrays.asList(VERB).contains(w)) return 'V';
        if (Arrays.asList(PREP).contains(w)) return 'P';
        return '?';
    }

    static int size(char c) {
        switch (c) {
            case 'D': return DET.length;
            case 'N': return NOUN.length;
            case 'A': return ADJ.length;
            case 'V': return VERB.length;
            case 'P': return PREP.length;
            default: throw new IllegalArgumentException("catégorie " + c);
        }
    }

    static String cats(String sentence) {
        StringBuilder sb = new StringBuilder();
        for (String w : sentence.trim().split("\\s+")) sb.append(category(w));
        return sb.toString();
    }

    private static final Pattern GRAMMAR = Pattern.compile("DA?NVP?DA?N");

    /** Issue d'une phrase complétée. */
    public enum Outcome {
        VALID, INCOMPLETE /* valide jusqu'ici mais s'arrête trop tôt */, VIOLATION /* catégorie interdite ou trop long */
    }

    public static Outcome classify(String sentence) {
        Matcher m = GRAMMAR.matcher(cats(sentence));
        if (m.matches()) return Outcome.VALID;
        return m.hitEnd() ? Outcome.INCOMPLETE : Outcome.VIOLATION;
    }

    /** Les 8 structures possibles (adjectif 1, préposition, adjectif 2), équiprobables. */
    private static List<String> structures() {
        List<String> out = new ArrayList<>();
        for (int a1 = 0; a1 < 2; a1++) for (int p = 0; p < 2; p++) for (int a2 = 0; a2 < 2; a2++) {
            out.add("D" + (a1 == 1 ? "A" : "") + "NV" + (p == 1 ? "P" : "") + "D" + (a2 == 1 ? "A" : "") + "N");
        }
        return out;
    }

    /**
     * Posterior EXACT des structures sachant l'amorce : la longueur d'amorce k est tirée
     * uniformément dans 1..n-1, donc P(struct | amorce) ∝ 1/(n-1) pour les structures
     * compatibles (n > k). Retourne {structure -> probabilité}.
     */
    static java.util.Map<String, Double> posterior(String prefixCats) {
        java.util.Map<String, Double> w = new java.util.LinkedHashMap<>();
        double z = 0;
        for (String s : structures()) {
            if (s.length() > prefixCats.length() && s.startsWith(prefixCats)) {
                double x = 1.0 / (s.length() - 1);
                w.put(s, x);
                z += x;
            }
        }
        for (java.util.Map.Entry<String, Double> e : w.entrySet()) e.setValue(e.getValue() / z);
        return w;
    }

    private static double lnSizes(String structure, int from) {
        double s = 0;
        for (int i = from; i < structure.length(); i++) s += Math.log(size(structure.charAt(i)));
        return s;
    }

    /** NLL (nats) de l'ORACLE sur la suite réelle, tokens = mots de la suite + END. */
    public static double oracleNll(Item it) {
        String pc = cats(it.prefix);
        String full = cats(it.full());
        double post = posterior(pc).get(full);
        return -Math.log(post) + lnSizes(full, pc.length());
    }

    public static int tokenCount(Item it) {
        return it.continuation.split(" ").length + 1; // + END
    }

    /** Probabilité de la suite la plus probable : plafond de la correspondance exacte (décodage déterministe). */
    public static double modeProbability(Item it) {
        String pc = cats(it.prefix);
        double best = 0;
        for (java.util.Map.Entry<String, Double> e : posterior(pc).entrySet()) {
            best = Math.max(best, e.getValue() * Math.exp(-lnSizes(e.getKey(), pc.length())));
        }
        return best;
    }

    /** Plancher théorique (nats/token) de la loi d'entraînement : E[H(suite|amorce)] / E[#tokens]. */
    public static double populationLossFloor() {
        double totH = 0, totTok = 0;
        java.util.Map<String, double[]> groups = new java.util.LinkedHashMap<>();
        for (String s : structures()) {
            int n = s.length();
            for (int k = 1; k < n; k++) {
                groups.computeIfAbsent(s.substring(0, k), x -> new double[1])[0] += (1.0 / 8) * (1.0 / (n - 1));
            }
        }
        for (java.util.Map.Entry<String, double[]> g : groups.entrySet()) {
            String c = g.getKey();
            double p = g.getValue()[0];
            java.util.Map<String, Double> post = posterior(c);
            double hs = 0, hw = 0, tok = 0;
            for (java.util.Map.Entry<String, Double> e : post.entrySet()) {
                double q = e.getValue();
                hs -= q * Math.log(q);
                hw += q * lnSizes(e.getKey(), c.length());
                tok += q * (e.getKey().length() - c.length() + 1);
            }
            totH += p * (hs + hw);
            totTok += p * tok;
        }
        return totH / totTok;
    }

    // ------------------------------------------------------------ statistiques
    public static double[] wilson(int k, int n) {
        double z = 1.96, ph = (double) k / n, d = 1 + z * z / n, c = ph + z * z / (2 * n);
        double a = z * Math.sqrt(ph * (1 - ph) / n + z * z / (4.0 * n * n));
        return new double[] { (c - a) / d, (c + a) / d };
    }

    /** p-value bilatérale exacte du test des signes (appariés) : b améliorations vs c dégradations. */
    public static double signTestP(int b, int c) {
        int n = b + c;
        if (n == 0) return 1.0;
        int k = Math.min(b, c);
        double p = 0;
        for (int i = 0; i <= k; i++) p += binomial(n, i) * Math.pow(0.5, n);
        return Math.min(1.0, 2 * p);
    }

    private static double binomial(int n, int k) {
        double r = 1;
        for (int i = 1; i <= k; i++) r = r * (n - k + i) / i;
        return r;
    }

    // ------------------------------------------------------------- évaluation
    public static class EvalResult {
        public String label;
        public int n, valid, incomplete, violation, exact;
        public double nll, oracleNll, modeCeiling;
        public boolean[] validFlags;
        public String[] outputs;

        public double validity() { return valid / (double) n; }
        public double excessNll() { return nll - oracleNll; }

        @Override
        public String toString() {
            double[] ci = wilson(valid, n);
            return String.format("%s n=%d | valide=%d (%.1f%%, IC95 %.1f–%.1f) incomplète=%d violation=%d | "
                    + "exact=%d (%.1f%%, plafond %.1f%%) | NLL=%.4f oracle=%.4f excès=%.4f",
                    label, n, valid, 100.0 * valid / n, 100 * ci[0], 100 * ci[1], incomplete, violation,
                    exact, 100.0 * exact / n, 100 * modeCeiling, nll, oracleNll, excessNll());
        }
    }

    static String norm(String s) {
        return s == null ? "" : s.replaceAll("<[A-Z]+>", " ").trim().replaceAll("\\s+", " ");
    }

    /** Évaluation déterministe (décodage glouton : température 0,05, top-1). */
    public static EvalResult evaluate(TransformerModel model, Tokenizer tok, List<Item> items, String label) {
        EvalResult r = new EvalResult();
        r.label = label;
        r.n = items.size();
        r.validFlags = new boolean[r.n];
        r.outputs = new String[r.n];
        double modeSum = 0, oracleSum = 0;
        long oracleTokens = 0;
        for (int i = 0; i < r.n; i++) {
            Item it = items.get(i);
            String cont = norm(model.generate(it.prefix, 8, 0.05, 1, 7L));
            r.outputs[i] = cont;
            Outcome o = cont.isEmpty() ? Outcome.INCOMPLETE : classify(it.prefix + " " + cont);
            if (o == Outcome.VALID) { r.valid++; r.validFlags[i] = true; }
            else if (o == Outcome.INCOMPLETE) r.incomplete++;
            else r.violation++;
            if (cont.equals(it.continuation)) r.exact++;
            modeSum += modeProbability(it);
            oracleSum += oracleNll(it);
            oracleTokens += tokenCount(it);
        }
        r.modeCeiling = modeSum / r.n;
        r.oracleNll = oracleSum / oracleTokens;
        List<String> in = new ArrayList<>(), out = new ArrayList<>();
        for (Item it : items) { in.add(it.prefix); out.add(it.continuation); }
        r.nll = model.evaluateLoss(new DataGenerator(in, out, tok, 16, MAX_LEN));
        return r;
    }
}
