package RN.transformer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.junit.Test;

import RN.transformer.CorpusV2.Violation;

/** Le corpus v2 est exact : générateur, validateurs à 3 niveaux, oracle, jeux disjoints. */
public class CorpusV2Test {

    private static CorpusV2.Check chk(String s) { return CorpusV2.check(s.split(" ")); }

    private static void expect(String s, boolean l1, boolean l2, boolean l3, Violation first) {
        CorpusV2.Check c = chk(s);
        String msg = "[" + s + "] ";
        assertEquals(msg + "N1", l1, c.l1);
        assertEquals(msg + "N2", l2, c.l2);
        assertEquals(msg + "N3", l3, c.l3);
        assertEquals(msg + "première violation (toutes=" + c.all + ")", first, c.first);
    }

    @Test
    public void testSpecProbabilitiesAreNormalised() {
        for (int v = 0; v < CorpusV2.V; v++) {
            double s = 0, c = 0;
            for (CorpusV2.NpSpec sp : CorpusV2.npSpecs(CorpusV2.VERBS[v].subj)) s += sp.p;
            for (CorpusV2.CompSpec cs : CorpusV2.compSpecs(v)) c += cs.p;
            assertEquals("sujet " + CorpusV2.VERBS[v].sg, 1.0, s, 1e-9);
            assertEquals("complément " + CorpusV2.VERBS[v].sg, 1.0, c, 1e-9);
        }
    }

    @Test
    public void testEverySampledSentenceIsFullyValid() {
        Random r = new Random(1);
        for (int i = 0; i < 20000; i++) {
            String[] s = CorpusV2.sample(r);
            CorpusV2.Check c = CorpusV2.check(s);
            assertTrue(String.join(" ", s) + " -> " + c.all, c.l3);
            assertTrue(CorpusV2.pGen(s) > 0);
            assertTrue("longueur " + s.length, s.length >= 3 && s.length <= 8);
        }
    }

    @Test
    public void testSamplerFollowsStatedDistribution() {
        Random r = new Random(2);
        int n = 60000, sg = 0;
        int[] verb = new int[CorpusV2.V];
        for (int i = 0; i < n; i++) {
            String[] s = CorpusV2.sample(r);
            CorpusV2.Parse p = CorpusV2.parse(s);
            verb[CorpusV2.VERB_OF.get(s[p.iVerb])]++;
            boolean plural = "les".equals(s[0]) || "des".equals(s[0]);
            if (!plural) sg++;
        }
        for (int c : verb) assertEquals(1.0 / CorpusV2.V, c / (double) n, 0.006);
        assertEquals(CorpusV2.P_SG, sg / (double) n, 0.01);
    }

    @Test
    public void testAgreementViolationsAreDetectedAtLevelTwo() {
        expect("le chat mange la pomme", true, true, true, null);
        expect("les chats mangent les pommes", true, true, true, null);
        expect("le chat mangent la pomme", true, false, false, Violation.SUBJECT_VERB_AGREEMENT);
        expect("les chats mange la pomme", true, false, false, Violation.SUBJECT_VERB_AGREEMENT);
        expect("la chat mange la pomme", true, false, false, Violation.DET_NOUN_AGREEMENT);
        expect("le chat mange la pommes", true, false, false, Violation.DET_NOUN_AGREEMENT);
        expect("le chat petite mange la pomme", true, false, false, Violation.ADJ_NOUN_AGREEMENT);
        expect("la femme heureuse habite dans la maison", true, true, true, null);
        expect("la femme heureux habite dans la maison", true, false, false, Violation.ADJ_NOUN_AGREEMENT);
        // « souris » : même forme au singulier et au pluriel, le déterminant tranche
        expect("les souris mangent le pain", true, true, true, null);
        expect("la souris mange le pain", true, true, true, null);
        expect("les souris mange le pain", true, false, false, Violation.SUBJECT_VERB_AGREEMENT);
        expect("la souris mangent le pain", true, false, false, Violation.SUBJECT_VERB_AGREEMENT);
    }

    @Test
    public void testSubcategorisation() {
        expect("le chat mange", true, false, false, Violation.SUBCATEGORIZATION);
        expect("le chat dort la pomme", true, false, false, Violation.SUBCATEGORIZATION);
        expect("le chat habite", true, false, false, Violation.SUBCATEGORIZATION);
        expect("le chat dort", true, true, true, null);
        expect("le chat habite dans la maison", true, true, true, null);
        expect("le chat mange dans la maison", true, false, false, Violation.SUBCATEGORIZATION);
    }

    @Test
    public void testSemanticViolationsAreDetectedOnlyAtLevelThree() {
        expect("la pomme mange le chat", true, true, false, Violation.SUBJECT_SEMANTICS);
        expect("le chat mange la table", true, true, false, Violation.OBJECT_SEMANTICS);
        expect("le chat lit la lettre", true, true, false, Violation.SUBJECT_SEMANTICS);
        expect("le garcon lit la lettre", true, true, true, null);
        expect("le chat frais mange la pomme", true, true, false, Violation.ADJECTIVE_SEMANTICS);
        expect("le chat mange la pomme fraiche", true, true, true, null);
        expect("le chat dort dans la maison", true, true, true, null);
        expect("le chat dort dans la pomme", true, true, false, Violation.LOCATIVE_SEMANTICS);
        expect("la voiture roule sous la maison", true, true, false, Violation.LOCATIVE_SEMANTICS);
        expect("la voiture roule dans la rue", true, true, true, null);
        expect("le chat dort sur la table", true, true, true, null);
        expect("la voiture roule sur la table", true, true, false, Violation.LOCATIVE_SEMANTICS);
    }

    @Test
    public void testStructureUnknownAndIncomplete() {
        expect("le mange chat", false, false, false, Violation.STRUCTURE);
        expect("le chat voit la maison dans", false, false, false, Violation.STRUCTURE);
        expect("le chat", false, false, false, Violation.INCOMPLETE);
        expect("le chat voit la", false, false, false, Violation.INCOMPLETE);
        expect("le chien mange la banane", false, false, false, Violation.UNKNOWN_TOKEN);
        assertEquals(Violation.INCOMPLETE, CorpusV2.check(new String[0]).first);
    }

    /** pGen > 0 <=> niveau N3, et pGen égale le produit des probabilités des spécifications (calcul indépendant). */
    @Test
    public void testGeneratorSupportEqualsLevelThree() {
        Random r = new Random(3);
        List<String> vocab = CorpusV2.vocabulary().subList(4, CorpusV2.vocabulary().size());
        int mutated = 0, valid = 0, checkedAgainstSpecs = 0;
        for (int i = 0; i < 40000; i++) {
            String[] s = CorpusV2.sample(r).clone();
            if (i % 2 == 0) s[r.nextInt(s.length)] = vocab.get(r.nextInt(vocab.size()));
            mutated++;
            CorpusV2.Check c = CorpusV2.check(s);
            double p = CorpusV2.pGen(s);
            assertEquals(String.join(" ", s), c.l3, p > 0);
            if (c.l3) {
                valid++;
                // probabilité indépendante : 1/V * P(sujet) * P(complément) retrouvés dans les listes de spécifications
                CorpusV2.Parse pr = CorpusV2.parse(s);
                int vi = CorpusV2.VERB_OF.get(s[pr.iVerb]);
                double ps = -1, pc = -1;
                for (CorpusV2.NpSpec sp : CorpusV2.npSpecs(CorpusV2.VERBS[vi].subj)) {
                    if (Arrays.equals(sp.tok, Arrays.copyOfRange(s, 0, pr.iVerb))) { ps = sp.p; break; }
                }
                String[] comp = Arrays.copyOfRange(s, pr.iVerb + 1, s.length);
                for (CorpusV2.CompSpec cs : CorpusV2.compSpecs(vi)) {
                    if (Arrays.equals(cs.tok, comp)) { pc = cs.p; break; }
                }
                assertTrue("phrase N3 absente des spécifications : " + String.join(" ", s), ps > 0 && pc > 0);
                assertEquals(String.join(" ", s), (1.0 / CorpusV2.V) * ps * pc, p, 1e-12);
                checkedAgainstSpecs++;
            }
        }
        assertTrue(valid > 10000 && valid < mutated);
        assertTrue(checkedAgainstSpecs == valid);
    }

    @Test
    public void testOracleNormalisationMatchesBruteForceEnumeration() {
        Random r = new Random(4);
        int tested = 0;
        while (tested < 40) {
            String[] s = CorpusV2.sample(r);
            int k = 4 + r.nextInt(Math.max(1, s.length - 4)); // préfixe long : énumération brute bornée
            if (k >= s.length) continue;
            String[] pre = Arrays.copyOfRange(s, 0, k);
            double zFast = CorpusV2.zAndMode(pre)[0];
            double zBrute = 0, modeBrute = 0;
            for (int vi = 0; vi < CorpusV2.V; vi++) {
                CorpusV2.Verb v = CorpusV2.VERBS[vi];
                for (CorpusV2.NpSpec sp : CorpusV2.npSpecs(v.subj)) {
                    int len = sp.tok.length;
                    if (!Arrays.equals(Arrays.copyOfRange(pre, 0, Math.min(k, len)), Arrays.copyOfRange(sp.tok, 0, Math.min(k, len)))) continue;
                    for (CorpusV2.CompSpec cs : CorpusV2.compSpecs(vi)) {
                        List<String> full = new ArrayList<>(Arrays.asList(sp.tok));
                        full.add(sp.num == 0 ? v.sg : v.pl);
                        full.addAll(Arrays.asList(cs.tok));
                        if (full.size() <= k) continue;
                        if (!full.subList(0, k).equals(Arrays.asList(pre))) continue;
                        double w = CorpusV2.pGen(full.toArray(new String[0])) / (full.size() - 1);
                        zBrute += w;
                        modeBrute = Math.max(modeBrute, w);
                    }
                }
            }
            assertEquals("Z factorisé vs énumération brute, préfixe " + String.join(" ", pre), zBrute, zFast, 1e-12 * Math.max(1, zBrute) + 1e-15);
            assertEquals(modeBrute, CorpusV2.zAndMode(pre)[1], 1e-12);
            tested++;
        }
    }

    /** Identité de normalisation : Σ_prefixes de longueur k Z(prefixe) = E[1{n>k}/(n-1)] (Monte-Carlo indépendant). */
    @Test
    public void testPrefixMassIdentity() {
        Random r = new Random(5);
        int m = 400000;
        double[] e = new double[3];
        for (int i = 0; i < m; i++) {
            int n = CorpusV2.sample(r).length;
            for (int k = 1; k <= 2; k++) if (n > k) e[k] += 1.0 / (n - 1);
        }
        for (int k = 1; k <= 2; k++) e[k] /= m;
        List<String> dets = Arrays.asList("le", "la", "les", "un", "une", "des");
        double sum1 = 0, sum2 = 0;
        List<String> nouns = new ArrayList<>();
        for (CorpusV2.Noun nn : CorpusV2.NOUNS) { nouns.add(nn.sg); if (!nn.pl.equals(nn.sg)) nouns.add(nn.pl); }
        for (String d : dets) {
            sum1 += CorpusV2.zAndMode(new String[] { d })[0];
            for (String nn : nouns) sum2 += CorpusV2.zAndMode(new String[] { d, nn })[0];
        }
        assertEquals("k=1", e[1], sum1, 0.004);
        assertEquals("k=2", e[2], sum2, 0.004);
    }

    @Test
    public void testOracleIsPositiveAndFloorIsStable() {
        double[] a = CorpusV2.populationLossFloor(6000, 11), b = CorpusV2.populationLossFloor(6000, 12);
        System.out.printf("CORPUSV2 plancher de perte = %.4f ± %.4f | %.4f ± %.4f (ln vocab = %.3f)%n", a[0], a[1], b[0], b[1],
                Math.log(CorpusV2.vocabulary().size()));
        assertTrue(a[0] > 0.5 && a[0] < Math.log(CorpusV2.vocabulary().size()));
        assertEquals(a[0], b[0], 4 * Math.hypot(a[1], b[1]));
        for (NextWordProtocol.Item it : CorpusV2.validationItems()) {
            double mode = CorpusV2.modeProbability(it);
            assertTrue(mode > 0 && mode <= 1.0 + 1e-12);
            assertTrue(CorpusV2.oracleNll(it) > 0);
        }
    }

    @Test
    public void testSplitsAreDisjointAndDynamicSamplingNeverLeaks() {
        Set<String> tr = new HashSet<>(CorpusV2.trainSentences()), va = new HashSet<>(CorpusV2.validationSentences()),
                te = new HashSet<>(CorpusV2.testSentences()), tf = new HashSet<>(CorpusV2.testFinalSentences());
        assertEquals(200, tr.size());
        assertTrue(CorpusV2.validationItems().size() >= 100 && CorpusV2.testItems().size() >= 100
                && CorpusV2.testFinalItems().size() >= 100);
        List<Set<String>> all = Arrays.asList(tr, va, te, tf);
        for (Set<String> a : all) for (Set<String> b : all) if (a != b) {
            Set<String> i = new HashSet<>(a);
            i.retainAll(b);
            assertTrue("fuite entre jeux : " + i, i.isEmpty());
        }
        for (NextWordProtocol.Item it : CorpusV2.testFinalItems()) assertTrue(tf.contains(it.full()));
        assertEquals(600, CorpusV2.trainItems().size());
        Set<String> forbidden = CorpusV2.forbiddenForTraining();
        for (int epoch = 1; epoch <= 15; epoch++) {
            NextWordProtocol.Sample a = CorpusV2.dynamicSample(42, epoch, 200, 3, forbidden);
            NextWordProtocol.Sample b = CorpusV2.dynamicSample(42, epoch, 200, 3, forbidden);
            assertEquals(600, a.items.size());
            Set<String> sentences = new HashSet<>();
            for (int i = 0; i < a.items.size(); i++) {
                assertEquals(a.items.get(i).full(), b.items.get(i).full());
                assertFalse("fuite : " + a.items.get(i).full(), forbidden.contains(a.items.get(i).full()));
                sentences.add(a.items.get(i).full());
                assertTrue(CorpusV2.check(a.items.get(i).full().split(" ")).l3);
            }
            assertEquals(200, sentences.size());
        }
    }

    @Test
    public void testVocabularyAndTokenizerAreConsistent() {
        List<String> v = CorpusV2.vocabulary();
        assertEquals(v.size(), new HashSet<>(v).size());
        assertNotNull(new Tokenizer(v, 16, CorpusV2.MAX_LEN));
        for (String s : CorpusV2.pool().subList(0, 200)) for (String w : s.split(" ")) assertTrue(w, v.contains(w));
        assertTrue(CorpusV2.MAX_LEN >= 8 + 2);
    }

    @Test
    public void printCorpusProfile() {
        Random r = new Random(7);
        int[] len = new int[10];
        int pp = 0, obj = 0, none = 0, adj = 0, n = 20000;
        Set<String> distinct = new HashSet<>();
        for (int i = 0; i < n; i++) {
            String[] s = CorpusV2.sample(r);
            len[s.length]++;
            CorpusV2.Parse p = CorpusV2.parse(s);
            if (p.kind == 2) pp++; else if (p.kind == 1) obj++; else none++;
            if (p.iAdj1 >= 0 || p.iAdj2 >= 0) adj++;
            distinct.add(String.join(" ", s));
        }
        System.out.println("CORPUSV2 vocabulaire=" + CorpusV2.vocabulary().size() + " (dont 4 spéciaux), noms=" + CorpusV2.NOUNS.length
                + ", adjectifs=" + CorpusV2.ADJS.length + ", verbes=" + CorpusV2.V);
        System.out.println("CORPUSV2 longueurs " + Arrays.toString(len) + " | objet=" + obj + " PP=" + pp + " sans complément=" + none
                + " | avec adjectif=" + adj + " | distinctes sur " + n + " tirages=" + distinct.size());
        Random e = new Random(8);
        for (int i = 0; i < 14; i++) System.out.println("CORPUSV2 exemple : " + String.join(" ", CorpusV2.sample(e)));
    }
}
