package RN.transformer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Corpus v2 : phrases françaises avec ACCORDS (genre/nombre) et CONTRAINTES SÉMANTIQUES.
 * Le corpus v1 (NextWordProtocol) est conservé intact.
 *
 * Structure :   SUJET VERBE [COMPLÉMENT]
 *   SUJET      = DET NOM [ADJ]                       (adjectif postposé, facultatif, p = 0,5)
 *   COMPLÉMENT = selon le cadre du verbe :
 *       T (transitif)    : objet  DET NOM [ADJ]                   (obligatoire)
 *       I (intransitif) : rien (p = 0,5) ou groupe prépositionnel  PREP DET NOM [ADJ]
 *       L (locatif)      : groupe prépositionnel                   (obligatoire)
 *
 * Contraintes (chacune est un NIVEAU de validité distinct, cumulatif) :
 *   N1 STRUCTURE     : patron de catégories D N A? V (D N A? | P D N A?)?
 *   N2 ACCORD        : déterminant–nom et adjectif–nom (genre + nombre), verbe–sujet (nombre),
 *                      sous-catégorisation (T exige un objet, L un groupe prépositionnel, I pas d'objet)
 *   N3 SÉMANTIQUE    : restrictions de sélection (classe du sujet, de l'objet, du lieu ; adjectif
 *                      compatible avec la classe du nom ; préposition compatible avec la classe du lieu)
 *
 * Histoire générative (probabilités exactes, utilisées par l'oracle) :
 *   verbe uniforme (11) -> sujet : nom uniforme parmi les classes admises, nombre (sg 0,6 / pl 0,4),
 *   déterminant uniforme parmi les 2 formes, adjectif absent (0,5) ou uniforme parmi les compatibles ->
 *   complément selon le cadre (préposition uniforme parmi celles compatibles, puis nom, etc.).
 * Les phrases générées = EXACTEMENT les phrases de niveau N3 (support de la loi) : pGen(s) > 0 <=> N3.
 *
 * Tâche d'entraînement identique à v1 : (début de phrase de k mots, k uniforme 1..n-1) -> (suite).
 */
public class CorpusV2 {

    public static final int MAX_LEN = 12;
    static final double P_SG = 0.6, P_PL = 0.4, P_ADJ = 0.5;

    public enum Cls { HUMAN, ANIMAL, FOOD, READABLE, VEHICLE, PLACE, OBJECT }
    public enum Frame { T, I, L }

    // ------------------------------------------------------------------ lexique
    static class Noun {
        final String sg, pl; final boolean fem; final Cls cls;
        Noun(String sg, String pl, boolean fem, Cls cls) { this.sg = sg; this.pl = pl; this.fem = fem; this.cls = cls; }
    }

    private static Noun n(String sg, String pl, boolean fem, Cls c) { return new Noun(sg, pl, fem, c); }

    static final Noun[] NOUNS = {
        n("enfant", "enfants", false, Cls.HUMAN), n("homme", "hommes", false, Cls.HUMAN),
        n("femme", "femmes", true, Cls.HUMAN), n("fille", "filles", true, Cls.HUMAN),
        n("garcon", "garcons", false, Cls.HUMAN),
        n("chat", "chats", false, Cls.ANIMAL), n("chien", "chiens", false, Cls.ANIMAL),
        n("oiseau", "oiseaux", false, Cls.ANIMAL), n("cheval", "chevaux", false, Cls.ANIMAL),
        n("souris", "souris", true, Cls.ANIMAL),
        n("pomme", "pommes", true, Cls.FOOD), n("pain", "pains", false, Cls.FOOD),
        n("gateau", "gateaux", false, Cls.FOOD), n("soupe", "soupes", true, Cls.FOOD),
        n("fromage", "fromages", false, Cls.FOOD),
        n("livre", "livres", false, Cls.READABLE), n("journal", "journaux", false, Cls.READABLE),
        n("lettre", "lettres", true, Cls.READABLE),
        n("voiture", "voitures", true, Cls.VEHICLE), n("train", "trains", false, Cls.VEHICLE),
        n("bateau", "bateaux", false, Cls.VEHICLE), n("velo", "velos", false, Cls.VEHICLE),
        n("maison", "maisons", true, Cls.PLACE), n("jardin", "jardins", false, Cls.PLACE),
        n("ville", "villes", true, Cls.PLACE), n("rue", "rues", true, Cls.PLACE),
        n("foret", "forets", true, Cls.PLACE),
        n("table", "tables", true, Cls.OBJECT), n("porte", "portes", true, Cls.OBJECT),
        n("fenetre", "fenetres", true, Cls.OBJECT),
    };

    static class Adj {
        final String ms, fs, mp, fp; final EnumSet<Cls> cls;
        Adj(String ms, String fs, String mp, String fp, EnumSet<Cls> cls) {
            this.ms = ms; this.fs = fs; this.mp = mp; this.fp = fp; this.cls = cls;
        }
        String form(boolean fem, boolean pl) { return pl ? (fem ? fp : mp) : (fem ? fs : ms); }
    }

    private static final EnumSet<Cls> ALL = EnumSet.allOf(Cls.class);
    private static EnumSet<Cls> cs(Cls... c) { return EnumSet.copyOf(Arrays.asList(c)); }

    static final Adj[] ADJS = {
        new Adj("petit", "petite", "petits", "petites", ALL),
        new Adj("grand", "grande", "grands", "grandes", ALL),
        new Adj("vieux", "vieille", "vieux", "vieilles", ALL),
        new Adj("gentil", "gentille", "gentils", "gentilles", cs(Cls.HUMAN, Cls.ANIMAL)),
        new Adj("heureux", "heureuse", "heureux", "heureuses", cs(Cls.HUMAN, Cls.ANIMAL)),
        new Adj("frais", "fraiche", "frais", "fraiches", cs(Cls.FOOD)),
        new Adj("rouge", "rouge", "rouges", "rouges", cs(Cls.FOOD, Cls.READABLE, Cls.VEHICLE, Cls.OBJECT)),
        new Adj("ancien", "ancienne", "anciens", "anciennes", cs(Cls.READABLE, Cls.VEHICLE, Cls.PLACE, Cls.OBJECT)),
        new Adj("rapide", "rapide", "rapides", "rapides", cs(Cls.HUMAN, Cls.ANIMAL, Cls.VEHICLE)),
        new Adj("calme", "calme", "calmes", "calmes", cs(Cls.HUMAN, Cls.ANIMAL, Cls.PLACE)),
    };

    static class Verb {
        final String sg, pl; final Frame frame; final EnumSet<Cls> subj, obj, loc;
        Verb(String sg, String pl, Frame f, EnumSet<Cls> subj, EnumSet<Cls> obj, EnumSet<Cls> loc) {
            this.sg = sg; this.pl = pl; this.frame = f; this.subj = subj; this.obj = obj; this.loc = loc;
        }
    }

    static final Verb[] VERBS = {
        new Verb("mange", "mangent", Frame.T, cs(Cls.HUMAN, Cls.ANIMAL), cs(Cls.FOOD), null),
        new Verb("lit", "lisent", Frame.T, cs(Cls.HUMAN), cs(Cls.READABLE), null),
        new Verb("conduit", "conduisent", Frame.T, cs(Cls.HUMAN), cs(Cls.VEHICLE), null),
        new Verb("voit", "voient", Frame.T, cs(Cls.HUMAN, Cls.ANIMAL), ALL, null),
        new Verb("ouvre", "ouvrent", Frame.T, cs(Cls.HUMAN), cs(Cls.OBJECT, Cls.READABLE), null),
        new Verb("traverse", "traversent", Frame.T, cs(Cls.HUMAN, Cls.ANIMAL, Cls.VEHICLE), cs(Cls.PLACE), null),
        new Verb("dort", "dorment", Frame.I, cs(Cls.HUMAN, Cls.ANIMAL), null, cs(Cls.PLACE, Cls.OBJECT)),
        new Verb("court", "courent", Frame.I, cs(Cls.HUMAN, Cls.ANIMAL), null, cs(Cls.PLACE, Cls.OBJECT)),
        new Verb("roule", "roulent", Frame.I, cs(Cls.VEHICLE), null, cs(Cls.PLACE)),
        new Verb("habite", "habitent", Frame.L, cs(Cls.HUMAN, Cls.ANIMAL), null, cs(Cls.PLACE)),
        new Verb("vit", "vivent", Frame.L, cs(Cls.HUMAN, Cls.ANIMAL), null, cs(Cls.PLACE)),
    };
    static final int V = VERBS.length;

    static class Prep {
        final String w; final EnumSet<Cls> cls;
        Prep(String w, EnumSet<Cls> cls) { this.w = w; this.cls = cls; }
    }

    static final Prep[] PREPS = {
        new Prep("dans", cs(Cls.PLACE, Cls.VEHICLE)),
        new Prep("sur", cs(Cls.OBJECT, Cls.PLACE, Cls.VEHICLE)),
        new Prep("devant", cs(Cls.PLACE, Cls.OBJECT, Cls.VEHICLE)),
        new Prep("derriere", cs(Cls.PLACE, Cls.OBJECT, Cls.VEHICLE)),
        new Prep("sous", cs(Cls.OBJECT, Cls.VEHICLE)),
    };

    /** Déterminants uniformes parmi 2 formes selon (genre, nombre). */
    static String[] dets(boolean fem, boolean pl) {
        if (pl) return new String[] { "les", "des" };
        return fem ? new String[] { "la", "une" } : new String[] { "le", "un" };
    }

    // ---------------------------------------------------- index token -> catégorie
    static final Map<String, Character> CAT = new HashMap<>();
    static final Map<String, Integer> NOUN_OF = new HashMap<>(), ADJ_OF = new HashMap<>(), VERB_OF = new HashMap<>(),
            PREP_OF = new HashMap<>();

    private static void reg(String tok, char cat) {
        Character old = CAT.put(tok, cat);
        if (old != null && old != cat) throw new IllegalStateException("token ambigu entre catégories : " + tok);
    }

    static {
        for (String d : new String[] { "le", "la", "les", "un", "une", "des" }) reg(d, 'D');
        for (int i = 0; i < NOUNS.length; i++) {
            reg(NOUNS[i].sg, 'N'); reg(NOUNS[i].pl, 'N');
            NOUN_OF.put(NOUNS[i].sg, i); NOUN_OF.put(NOUNS[i].pl, i);
        }
        for (int i = 0; i < ADJS.length; i++) {
            for (String f : new String[] { ADJS[i].ms, ADJS[i].fs, ADJS[i].mp, ADJS[i].fp }) {
                reg(f, 'A'); ADJ_OF.put(f, i);
            }
        }
        for (int i = 0; i < V; i++) {
            reg(VERBS[i].sg, 'V'); reg(VERBS[i].pl, 'V');
            VERB_OF.put(VERBS[i].sg, i); VERB_OF.put(VERBS[i].pl, i);
        }
        for (int i = 0; i < PREPS.length; i++) { reg(PREPS[i].w, 'P'); PREP_OF.put(PREPS[i].w, i); }
    }

    public static List<String> vocabulary() {
        List<String> v = new ArrayList<>(Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>"));
        v.addAll(new LinkedHashSet<>(CAT.keySet().stream().sorted().collect(java.util.stream.Collectors.toList())));
        return v;
    }

    // ------------------------------------------------- spécifications énumérables
    /** Syntagme nominal DET NOM [ADJ] et sa probabilité sachant l'ensemble de classes admis. */
    static class NpSpec {
        final String[] tok; final int num; final boolean adj; final double p; final int noun;
        NpSpec(String[] tok, int num, boolean adj, double p, int noun) {
            this.tok = tok; this.num = num; this.adj = adj; this.p = p; this.noun = noun;
        }
    }

    private static final Map<String, List<NpSpec>> NP_CACHE = new HashMap<>();

    static List<Integer> compatibleAdjs(Cls c) {
        List<Integer> out = new ArrayList<>();
        for (int a = 0; a < ADJS.length; a++) if (ADJS[a].cls.contains(c)) out.add(a);
        return out;
    }

    static synchronized List<NpSpec> npSpecs(Set<Cls> allowed) {
        String key = new java.util.TreeSet<>(allowed).toString();
        List<NpSpec> cached = NP_CACHE.get(key);
        if (cached != null) return cached;
        List<Integer> nouns = new ArrayList<>();
        for (int i = 0; i < NOUNS.length; i++) if (allowed.contains(NOUNS[i].cls)) nouns.add(i);
        List<NpSpec> out = new ArrayList<>();
        for (int ni : nouns) {
            Noun nn = NOUNS[ni];
            List<Integer> adjs = compatibleAdjs(nn.cls);
            for (int num = 0; num < 2; num++) {
                boolean pl = num == 1;
                for (String d : dets(nn.fem, pl)) {
                    double base = (1.0 / nouns.size()) * (pl ? P_PL : P_SG) * 0.5;
                    String noun = pl ? nn.pl : nn.sg;
                    out.add(new NpSpec(new String[] { d, noun }, num, false, base * (1 - P_ADJ), ni));
                    for (int a : adjs) {
                        out.add(new NpSpec(new String[] { d, noun, ADJS[a].form(nn.fem, pl) }, num, true,
                                base * P_ADJ / adjs.size(), ni));
                    }
                }
            }
        }
        NP_CACHE.put(key, out);
        return out;
    }

    /** Complément : NONE (0), OBJ (1) ou PP (2). */
    static class CompSpec {
        final String[] tok; final int kind; final double p;
        CompSpec(String[] tok, int kind, double p) { this.tok = tok; this.kind = kind; this.p = p; }
    }

    private static final Map<Integer, List<CompSpec>> COMP_CACHE = new HashMap<>();

    static List<Integer> validPreps(Verb v) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < PREPS.length; i++) {
            EnumSet<Cls> inter = EnumSet.copyOf(v.loc);
            inter.retainAll(PREPS[i].cls);
            if (!inter.isEmpty()) out.add(i);
        }
        return out;
    }

    static synchronized List<CompSpec> compSpecs(int vi) {
        List<CompSpec> cached = COMP_CACHE.get(vi);
        if (cached != null) return cached;
        Verb v = VERBS[vi];
        List<CompSpec> out = new ArrayList<>();
        if (v.frame == Frame.T) {
            for (NpSpec np : npSpecs(v.obj)) out.add(new CompSpec(np.tok, 1, np.p));
        } else {
            double ppMass = v.frame == Frame.I ? 0.5 : 1.0;
            if (v.frame == Frame.I) out.add(new CompSpec(new String[0], 0, 0.5));
            List<Integer> preps = validPreps(v);
            for (int pi : preps) {
                EnumSet<Cls> inter = EnumSet.copyOf(v.loc);
                inter.retainAll(PREPS[pi].cls);
                for (NpSpec np : npSpecs(inter)) {
                    String[] tok = new String[np.tok.length + 1];
                    tok[0] = PREPS[pi].w;
                    System.arraycopy(np.tok, 0, tok, 1, np.tok.length);
                    out.add(new CompSpec(tok, 2, ppMass * (1.0 / preps.size()) * np.p));
                }
            }
        }
        COMP_CACHE.put(vi, out);
        return out;
    }

    // ---------------------------------------------------------------- échantillonnage
    private static <T> T pickWeighted(Random r, List<T> items, double[] weights) {
        double u = r.nextDouble(), acc = 0;
        for (int i = 0; i < items.size(); i++) {
            acc += weights[i];
            if (u < acc) return items.get(i);
        }
        return items.get(items.size() - 1);
    }

    /** Tire une phrase selon l'histoire générative exacte. */
    public static String[] sample(Random r) {
        int vi = r.nextInt(V);
        Verb v = VERBS[vi];
        List<NpSpec> subj = npSpecs(v.subj);
        NpSpec s = drawNp(r, subj);
        List<String> out = new ArrayList<>(Arrays.asList(s.tok));
        out.add(s.num == 0 ? v.sg : v.pl);
        List<CompSpec> comps = compSpecs(vi);
        double[] w = new double[comps.size()];
        for (int i = 0; i < w.length; i++) w[i] = comps.get(i).p;
        CompSpec c = pickWeighted(r, comps, w);
        out.addAll(Arrays.asList(c.tok));
        return out.toArray(new String[0]);
    }

    private static NpSpec drawNp(Random r, List<NpSpec> specs) {
        double[] w = new double[specs.size()];
        for (int i = 0; i < w.length; i++) w[i] = specs.get(i).p;
        return pickWeighted(r, specs, w);
    }

    // ---------------------------------------------------------------- validation
    public enum Violation {
        UNKNOWN_TOKEN, STRUCTURE, INCOMPLETE,
        DET_NOUN_AGREEMENT, ADJ_NOUN_AGREEMENT, SUBJECT_VERB_AGREEMENT, SUBCATEGORIZATION,
        SUBJECT_SEMANTICS, OBJECT_SEMANTICS, ADJECTIVE_SEMANTICS, LOCATIVE_SEMANTICS
    }

    public static class Check {
        public boolean l1, l2, l3;
        public Violation first;
        public EnumSet<Violation> all = EnumSet.noneOf(Violation.class);
    }

    private static final Pattern GRAMMAR_FULL = Pattern.compile("DNA?V(DNA?|PDNA?)?");

    private static String cats(String[] t) {
        StringBuilder sb = new StringBuilder();
        for (String w : t) sb.append(CAT.getOrDefault(w, '?'));
        return sb.toString();
    }

    private static int detFeat(String d) {
        switch (d) {
            case "le": case "un": return 1 << 0;            // (M, sg)
            case "la": case "une": return 1 << 2;           // (F, sg)
            case "les": case "des": return (1 << 1) | (1 << 3); // (M, pl) | (F, pl)
            default: return 0;
        }
    }

    private static int nounFeat(String tok) {
        Noun nn = NOUNS[NOUN_OF.get(tok)];
        int g = nn.fem ? 2 : 0, f = 0;
        if (tok.equals(nn.sg)) f |= 1 << g;          // sg : code g + 0
        if (tok.equals(nn.pl)) f |= 1 << (g + 1);    // pl : code g + 1
        return f;
    }

    private static int adjFeat(String tok) {
        Adj a = ADJS[ADJ_OF.get(tok)];
        int f = 0;
        if (tok.equals(a.ms)) f |= 1 << 0;
        if (tok.equals(a.mp)) f |= 1 << 1;
        if (tok.equals(a.fs)) f |= 1 << 2;
        if (tok.equals(a.fp)) f |= 1 << 3;
        return f;
    }

    /** Positions (indices) des constituants d'une phrase de structure valide. */
    static class Parse {
        int iDet1 = 0, iNoun1 = 1, iAdj1 = -1, iVerb, kind, iPrep = -1, iDet2 = -1, iNoun2 = -1, iAdj2 = -1;
    }

    static Parse parse(String[] t) {
        String c = cats(t);
        if (!GRAMMAR_FULL.matcher(c).matches()) return null;
        Parse p = new Parse();
        int i = 2;
        if (c.charAt(i) == 'A') { p.iAdj1 = i; i++; }
        p.iVerb = i++;
        if (i == c.length()) { p.kind = 0; return p; }
        if (c.charAt(i) == 'P') { p.kind = 2; p.iPrep = i++; } else { p.kind = 1; }
        p.iDet2 = i++;
        p.iNoun2 = i++;
        if (i < c.length()) p.iAdj2 = i;
        return p;
    }

    public static Check check(String[] t) {
        Check r = new Check();
        for (String w : t) if (!CAT.containsKey(w)) { r.all.add(Violation.UNKNOWN_TOKEN); }
        if (t.length == 0) { r.all.add(Violation.INCOMPLETE); r.first = Violation.INCOMPLETE; return r; }
        String c = cats(t);
        Parse p = parse(t);
        if (p == null) {
            Matcher m = GRAMMAR_FULL.matcher(c);
            boolean incomplete = !r.all.contains(Violation.UNKNOWN_TOKEN) && !m.matches() && m.hitEnd();
            r.all.add(incomplete ? Violation.INCOMPLETE : Violation.STRUCTURE);
            r.first = r.all.contains(Violation.UNKNOWN_TOKEN) ? Violation.UNKNOWN_TOKEN
                    : (incomplete ? Violation.INCOMPLETE : Violation.STRUCTURE);
            return r;
        }
        r.l1 = true;
        Verb v = VERBS[VERB_OF.get(t[p.iVerb])];
        int verbNum = t[p.iVerb].equals(v.pl) && !v.sg.equals(v.pl) ? 1 : 0;

        // ---- N2 : accord
        int f1 = detFeat(t[p.iDet1]) & nounFeat(t[p.iNoun1]);
        if (f1 == 0) r.all.add(Violation.DET_NOUN_AGREEMENT);
        int fSubj = f1;
        if (p.iAdj1 >= 0) {
            fSubj = f1 & adjFeat(t[p.iAdj1]);
            if (f1 != 0 && fSubj == 0) r.all.add(Violation.ADJ_NOUN_AGREEMENT);
        }
        if (fSubj != 0) {
            boolean verbOk = false;
            for (int code = 0; code < 4; code++) {
                if ((fSubj & (1 << code)) != 0 && (code & 1) == verbNum) verbOk = true;
            }
            if (!verbOk) r.all.add(Violation.SUBJECT_VERB_AGREEMENT);
        }
        if (p.kind != 0) {
            int f2 = detFeat(t[p.iDet2]) & nounFeat(t[p.iNoun2]);
            if (f2 == 0) r.all.add(Violation.DET_NOUN_AGREEMENT);
            if (p.iAdj2 >= 0 && f2 != 0 && (f2 & adjFeat(t[p.iAdj2])) == 0) r.all.add(Violation.ADJ_NOUN_AGREEMENT);
        }
        boolean subcatOk = (v.frame == Frame.T && p.kind == 1) || (v.frame == Frame.L && p.kind == 2)
                || (v.frame == Frame.I && (p.kind == 0 || p.kind == 2));
        if (!subcatOk) r.all.add(Violation.SUBCATEGORIZATION);

        // ---- N3 : sémantique
        Noun n1 = NOUNS[NOUN_OF.get(t[p.iNoun1])];
        if (!v.subj.contains(n1.cls)) r.all.add(Violation.SUBJECT_SEMANTICS);
        if (p.iAdj1 >= 0 && !ADJS[ADJ_OF.get(t[p.iAdj1])].cls.contains(n1.cls)) r.all.add(Violation.ADJECTIVE_SEMANTICS);
        if (p.kind != 0) {
            Noun n2 = NOUNS[NOUN_OF.get(t[p.iNoun2])];
            if (p.iAdj2 >= 0 && !ADJS[ADJ_OF.get(t[p.iAdj2])].cls.contains(n2.cls)) r.all.add(Violation.ADJECTIVE_SEMANTICS);
            if (p.kind == 1 && v.frame == Frame.T && !v.obj.contains(n2.cls)) r.all.add(Violation.OBJECT_SEMANTICS);
            if (p.kind == 2 && v.loc != null) {
                Prep pr = PREPS[PREP_OF.get(t[p.iPrep])];
                if (!v.loc.contains(n2.cls) || !pr.cls.contains(n2.cls)) r.all.add(Violation.LOCATIVE_SEMANTICS);
            }
        }
        EnumSet<Violation> agree = EnumSet.of(Violation.DET_NOUN_AGREEMENT, Violation.ADJ_NOUN_AGREEMENT,
                Violation.SUBJECT_VERB_AGREEMENT, Violation.SUBCATEGORIZATION);
        EnumSet<Violation> sem = EnumSet.of(Violation.SUBJECT_SEMANTICS, Violation.OBJECT_SEMANTICS,
                Violation.ADJECTIVE_SEMANTICS, Violation.LOCATIVE_SEMANTICS);
        boolean agreeBad = !java.util.Collections.disjoint(r.all, agree);
        boolean semBad = !java.util.Collections.disjoint(r.all, sem);
        r.l2 = !agreeBad && !r.all.contains(Violation.UNKNOWN_TOKEN);
        r.l3 = r.l2 && !semBad;
        for (Violation x : Violation.values()) if (r.all.contains(x)) { r.first = x; break; }
        return r;
    }

    // ------------------------------------------------------------------- oracle
    /** Probabilité de génération d'une phrase (0 si elle n'est pas de niveau N3). */
    public static double pGen(String[] t) {
        Check c = check(t);
        if (!c.l3) return 0;
        Parse p = parse(t);
        int vi = VERB_OF.get(t[p.iVerb]);
        Verb v = VERBS[vi];
        double prob = 1.0 / V;
        prob *= npProb(t, p.iDet1, p.iNoun1, p.iAdj1, v.subj);
        if (p.kind == 1) {
            prob *= npProb(t, p.iDet2, p.iNoun2, p.iAdj2, v.obj);
        } else if (p.kind == 2 || (p.kind == 0 && v.frame == Frame.I)) {
            double ppMass = v.frame == Frame.I ? 0.5 : 1.0;
            if (p.kind == 0) return prob * 0.5;
            List<Integer> preps = validPreps(v);
            int pi = PREP_OF.get(t[p.iPrep]);
            EnumSet<Cls> inter = EnumSet.copyOf(v.loc);
            inter.retainAll(PREPS[pi].cls);
            prob *= ppMass * (1.0 / preps.size()) * npProb(t, p.iDet2, p.iNoun2, p.iAdj2, inter);
        }
        return prob;
    }

    private static double npProb(String[] t, int iDet, int iNoun, int iAdj, Set<Cls> allowed) {
        Noun nn = NOUNS[NOUN_OF.get(t[iNoun])];
        int nAllowed = 0;
        for (Noun x : NOUNS) if (allowed.contains(x.cls)) nAllowed++;
        boolean pl = (detFeat(t[iDet]) & ((1 << 1) | (1 << 3))) != 0;
        double p = (1.0 / nAllowed) * (pl ? P_PL : P_SG) * 0.5;
        if (iAdj < 0) return p * (1 - P_ADJ);
        return p * P_ADJ / compatibleAdjs(nn.cls).size();
    }

    private static boolean matches(String[] p, int from, String[] tok, int count) {
        for (int i = 0; i < count; i++) if (!p[from + i].equals(tok[i])) return false;
        return true;
    }

    /**
     * {Z, mode} pour une amorce : Z = somme sur les phrases s prolongeant l'amorce (n > k) de pGen(s)/(n-1) ;
     * mode = max de pGen(s)/(n-1). La suite réelle c a pour probabilité conditionnelle
     * pGen(amorce+c)/(n-1)/Z : l'amorce k ~ U{1..n-1} rend le postérieur sur la longueur ∝ 1/(n-1).
     * Calcul exact par factorisation (sujet et complément indépendants sachant le verbe).
     */
    public static double[] zAndMode(String[] p) {
        int k = p.length;
        double z = 0, best = 0;
        for (int vi = 0; vi < V; vi++) {
            Verb v = VERBS[vi];
            double[][] as = new double[2][2], am = new double[2][2];
            for (NpSpec sp : npSpecs(v.subj)) {
                int len = sp.tok.length;
                if (!matches(p, 0, sp.tok, Math.min(k, len))) continue;
                if (k > len && !p[len].equals(sp.num == 0 ? v.sg : v.pl)) continue;
                int a1 = sp.adj ? 1 : 0;
                as[sp.num][a1] += sp.p;
                am[sp.num][a1] = Math.max(am[sp.num][a1], sp.p);
            }
            List<CompSpec> comps = compSpecs(vi);
            for (int a1 = 0; a1 < 2; a1++) {
                int start = 3 + a1;
                double[] bs = new double[8], bm = new double[8];
                for (CompSpec cs : comps) {
                    int n = start + cs.tok.length;
                    if (n <= k) continue;
                    int rem = k - start;
                    if (rem > 0 && !matches(p, start, cs.tok, rem)) continue;
                    bs[cs.tok.length] += cs.p;
                    bm[cs.tok.length] = Math.max(bm[cs.tok.length], cs.p);
                }
                for (int s = 0; s < 2; s++) {
                    if (as[s][a1] == 0) continue;
                    for (int len = 0; len < bs.length; len++) {
                        if (bs[len] == 0) continue;
                        double w = 1.0 / (start + len - 1);
                        z += (1.0 / V) * as[s][a1] * bs[len] * w;
                        best = Math.max(best, (1.0 / V) * am[s][a1] * bm[len] * w);
                    }
                }
            }
        }
        return new double[] { z, best };
    }

    /** NLL (nats) de l'oracle sur la suite réelle d'un exemple (tokens = mots de la suite + END). */
    public static double oracleNll(NextWordProtocol.Item it) {
        String[] pre = it.prefix.split(" ");
        String[] full = it.full().split(" ");
        double z = zAndMode(pre)[0];
        return -Math.log(pGen(full) / (full.length - 1) / z);
    }

    public static double modeProbability(NextWordProtocol.Item it) {
        double[] zm = zAndMode(it.prefix.split(" "));
        return zm[1] / zm[0];
    }

    public static int tokenCount(NextWordProtocol.Item it) {
        return it.continuation.split(" ").length + 1;
    }

    /** Plancher de perte (nats/token) de la loi d'entraînement, estimé par Monte-Carlo : {valeur, erreur-type}. */
    public static double[] populationLossFloor(int samples, long seed) {
        Random r = new Random(seed);
        double[] nll = new double[samples], tok = new double[samples];
        for (int i = 0; i < samples; i++) {
            String[] s = sample(r);
            int k = 1 + r.nextInt(s.length - 1);
            NextWordProtocol.Item it = new NextWordProtocol.Item(String.join(" ", Arrays.copyOfRange(s, 0, k)),
                    String.join(" ", Arrays.copyOfRange(s, k, s.length)));
            nll[i] = oracleNll(it);
            tok[i] = tokenCount(it);
        }
        double mn = 0, mt = 0;
        for (int i = 0; i < samples; i++) { mn += nll[i]; mt += tok[i]; }
        mn /= samples; mt /= samples;
        // erreur-type du rapport des moyennes (méthode delta)
        double var = 0;
        for (int i = 0; i < samples; i++) { double d = nll[i] - (mn / mt) * tok[i]; var += d * d; }
        var /= (samples - 1);
        return new double[] { mn / mt, Math.sqrt(var / samples) / mt };
    }

    // ------------------------------------------------------------ jeux et échantillonnage
    public static List<String> pool() {
        Random r = new Random(20241);
        Set<String> all = new LinkedHashSet<>();
        while (all.size() < 1000) all.add(String.join(" ", sample(r)));
        return new ArrayList<>(all);
    }

    public static List<String> trainSentences() { return new ArrayList<>(pool().subList(0, 200)); }
    public static List<String> validationSentences() { return new ArrayList<>(pool().subList(200, 350)); }
    public static List<String> testSentences() { return new ArrayList<>(pool().subList(350, 550)); }
    public static List<String> testFinalSentences() { return new ArrayList<>(pool().subList(550, 750)); }
    public static Set<String> forbiddenForTraining() { return new HashSet<>(pool().subList(0, 750)); }

    private static NextWordProtocol.Item itemOf(String sentence, Random prng) {
        String[] w = sentence.split(" ");
        int k = 1 + prng.nextInt(w.length - 1);
        return new NextWordProtocol.Item(String.join(" ", Arrays.copyOfRange(w, 0, k)),
                String.join(" ", Arrays.copyOfRange(w, k, w.length)));
    }

    private static List<NextWordProtocol.Item> oneItemPerSentence(List<String> s, long seed) {
        Random prng = new Random(seed);
        List<NextWordProtocol.Item> out = new ArrayList<>();
        for (String x : s) out.add(itemOf(x, prng));
        return out;
    }

    public static List<NextWordProtocol.Item> trainItems() {
        Random prng = new Random(99);
        List<NextWordProtocol.Item> out = new ArrayList<>();
        for (String s : trainSentences()) for (int j = 0; j < 3; j++) out.add(itemOf(s, prng));
        return out;
    }

    public static List<NextWordProtocol.Item> validationItems() { return oneItemPerSentence(validationSentences(), 31337L); }
    public static List<NextWordProtocol.Item> testItems() { return oneItemPerSentence(testSentences(), 4242L); }
    public static List<NextWordProtocol.Item> testFinalItems() { return oneItemPerSentence(testFinalSentences(), 777L); }

    /** Échantillonnage dynamique : 200 phrases nouvelles par epoch (hors jeux), 3 amorces chacune. */
    public static NextWordProtocol.Sample dynamicSample(long seed, int epoch, int sentences, int perSentence,
            Set<String> forbidden) {
        Random r = new Random(seed * 1_000_003L + epoch);
        Set<String> chosen = new LinkedHashSet<>();
        while (chosen.size() < sentences) {
            String s = String.join(" ", sample(r));
            if (!forbidden.contains(s)) chosen.add(s);
        }
        NextWordProtocol.Sample out = new NextWordProtocol.Sample();
        for (String s : chosen) {
            for (int j = 0; j < perSentence; j++) {
                NextWordProtocol.Item it = itemOf(s, r);
                out.items.add(it);
                out.supervisedTokens += tokenCount(it);
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- évaluation
    public static class EvalV2 {
        public String label;
        public int n, l1, l2, l3, exact;
        public double nll, oracleNll, modeCeiling;
        public Map<Violation, Integer> firstViolations = new EnumMap<>(Violation.class);
        public boolean[] l3Flags;

        public String toString() {
            double[] ci = NextWordProtocol.wilson(l3, n);
            return String.format("%s n=%d | N1 structure=%d N2 accord=%d N3 sémantique=%d (%.1f%%, IC95 %.1f–%.1f) | "
                    + "exact=%d (plafond %.1f%%) | NLL=%.4f oracle=%.4f excès=%.4f | premières violations=%s",
                    label, n, l1, l2, l3, 100.0 * l3 / n, 100 * ci[0], 100 * ci[1], exact, 100 * modeCeiling,
                    nll, oracleNll, nll - oracleNll, firstViolations);
        }
    }

    /** Évaluation déterministe (glouton) : validité aux 3 niveaux, correspondance exacte, NLL vs oracle. */
    public static EvalV2 evaluate(TransformerModel model, Tokenizer tok, List<NextWordProtocol.Item> items, String label) {
        EvalV2 r = new EvalV2();
        r.label = label;
        r.n = items.size();
        r.l3Flags = new boolean[r.n];
        double modeSum = 0, oracleSum = 0;
        long oracleTok = 0;
        for (int i = 0; i < r.n; i++) {
            NextWordProtocol.Item it = items.get(i);
            String cont = NextWordProtocol.norm(model.generate(it.prefix, 8, 0.05, 1, 7L));
            String full = cont.isEmpty() ? it.prefix : it.prefix + " " + cont;
            Check c = check(full.split(" "));
            if (cont.isEmpty()) { c = new Check(); c.first = Violation.INCOMPLETE; }
            if (c.l1) r.l1++;
            if (c.l2) r.l2++;
            if (c.l3) { r.l3++; r.l3Flags[i] = true; }
            if (!c.l3 && c.first != null) r.firstViolations.merge(c.first, 1, Integer::sum);
            if (cont.equals(it.continuation)) r.exact++;
            modeSum += modeProbability(it);
            oracleSum += oracleNll(it);
            oracleTok += tokenCount(it);
        }
        r.modeCeiling = modeSum / r.n;
        r.oracleNll = oracleSum / oracleTok;
        List<String> in = new ArrayList<>(), out = new ArrayList<>();
        for (NextWordProtocol.Item it : items) { in.add(it.prefix); out.add(it.continuation); }
        r.nll = model.evaluateLoss(new DataGenerator(in, out, tok, 16, MAX_LEN));
        return r;
    }
}
