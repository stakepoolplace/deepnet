package RN.transformer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.Test;

/** Le protocole d'évaluation est fiable : jeux disjoints, déterministe, oracle exact. */
public class NextWordProtocolTest {

    @Test
    public void testSplitsAreDisjointAndSizedAsSpecified() {
        Set<String> train = new HashSet<>(NextWordProtocol.trainSentences());
        Set<String> old = new HashSet<>(NextWordProtocol.selectionOldSentences());
        Set<String> val = new HashSet<>(NextWordProtocol.validationSentences());
        Set<String> test = new HashSet<>(NextWordProtocol.testSentences());
        assertEquals(200, train.size());
        assertTrue("validation >= 100 amorces", NextWordProtocol.validationItems().size() >= 100);
        assertTrue("test >= 100 amorces", NextWordProtocol.testItems().size() >= 100);
        for (Set<String> a : java.util.Arrays.asList(train, old, val, test)) {
            for (Set<String> b : java.util.Arrays.asList(train, old, val, test)) {
                if (a != b) {
                    Set<String> inter = new HashSet<>(a);
                    inter.retainAll(b);
                    assertTrue("fuite de phrases entre jeux : " + inter, inter.isEmpty());
                }
            }
        }
        // Les amorces de validation/test sont des préfixes de LEURS phrases, pas de l'entraînement.
        for (NextWordProtocol.Item it : NextWordProtocol.validationItems()) {
            assertTrue(val.contains(it.full()));
        }
        for (NextWordProtocol.Item it : NextWordProtocol.testItems()) {
            assertTrue(test.contains(it.full()));
        }
    }

    @Test
    public void testTestFinalIsDisjointAndDynamicSamplingNeverLeaks() {
        Set<String> testFinal = new HashSet<>(NextWordProtocol.testFinalSentences());
        assertEquals(200, testFinal.size());
        for (List<String> other : java.util.Arrays.asList(NextWordProtocol.trainSentences(),
                NextWordProtocol.selectionOldSentences(), NextWordProtocol.validationSentences(),
                NextWordProtocol.testSentences())) {
            Set<String> inter = new HashSet<>(testFinal);
            inter.retainAll(other);
            assertTrue("TEST-FINAL recoupe un autre jeu : " + inter, inter.isEmpty());
        }
        Set<String> forbidden = NextWordProtocol.forbiddenForTraining();
        Set<String> seenAcrossEpochs = new HashSet<>();
        int repeats = 0;
        for (int epoch = 1; epoch <= 20; epoch++) {
            NextWordProtocol.Sample a = NextWordProtocol.dynamicSample(42, epoch, 200, 3, forbidden);
            NextWordProtocol.Sample b = NextWordProtocol.dynamicSample(42, epoch, 200, 3, forbidden);
            assertEquals("même (graine, epoch) => mêmes données", a.items.size(), b.items.size());
            for (int i = 0; i < a.items.size(); i++) assertEquals(a.items.get(i).full(), b.items.get(i).full());
            assertEquals(600, a.items.size());
            Set<String> sentences = new HashSet<>();
            for (NextWordProtocol.Item it : a.items) {
                assertTrue("fuite vers validation/test : " + it.full(), !forbidden.contains(it.full()));
                assertTrue("la phrase de TEST-FINAL ne doit jamais être échantillonnée", !testFinal.contains(it.full()));
                sentences.add(it.full());
            }
            assertEquals("200 phrases distinctes par epoch", 200, sentences.size());
            for (String x : sentences) if (!seenAcrossEpochs.add(x)) repeats++;
        }
        // l'espace grammatical (~25 M de phrases) rend les répétitions inter-epochs quasi nulles
        assertTrue("trop de phrases répétées entre epochs : " + repeats, repeats <= 2);
    }

    @Test
    public void testTrainingPairsAreTheOriginalCorpus() {
        List<NextWordProtocol.Item> items = NextWordProtocol.trainItems();
        assertEquals(600, items.size());
        Set<String> train = new HashSet<>(NextWordProtocol.trainSentences());
        for (NextWordProtocol.Item it : items) {
            assertTrue(train.contains(it.full()));
        }
    }

    @Test
    public void testDeterministic() {
        assertEquals(NextWordProtocol.pool(), NextWordProtocol.pool());
        List<NextWordProtocol.Item> a = NextWordProtocol.validationItems();
        List<NextWordProtocol.Item> b = NextWordProtocol.validationItems();
        for (int i = 0; i < a.size(); i++) {
            assertEquals(a.get(i).full(), b.get(i).full());
            assertEquals(a.get(i).prefix, b.get(i).prefix);
        }
    }

    @Test
    public void testPopulationLossFloorMatchesIndependentComputation() {
        // 1.9608 nats/token calculé indépendamment (Python, énumération exacte).
        assertEquals(1.9608, NextWordProtocol.populationLossFloor(), 1e-3);
    }

    @Test
    public void testOracleOnHandComputedCase() {
        // Amorce DNVPD : structures DNVPDN (n=6, poids 1/5) et DNVPDAN (n=7, poids 1/6).
        NextWordProtocol.Item it = new NextWordProtocol.Item("le chat voit dans la", "chien");
        double expected = -Math.log(6.0 / 11.0) + Math.log(15);
        assertEquals(expected, NextWordProtocol.oracleNll(it), 1e-9);
        assertEquals(2, NextWordProtocol.tokenCount(it)); // chien + END
        // Suite la plus probable : DNVPDN (6/11) avec un nom parmi 15.
        assertEquals(6.0 / 11.0 / 15.0, NextWordProtocol.modeProbability(it), 1e-9);
    }

    @Test
    public void testClassification() {
        assertEquals(NextWordProtocol.Outcome.VALID, NextWordProtocol.classify("le petit chat voit dans la maison"));
        assertEquals(NextWordProtocol.Outcome.VALID, NextWordProtocol.classify("une enfant aime la oiseau")); // sans sémantique
        assertEquals(NextWordProtocol.Outcome.INCOMPLETE, NextWordProtocol.classify("le chat voit"));
        assertEquals(NextWordProtocol.Outcome.VIOLATION, NextWordProtocol.classify("le chat voit voit"));
        assertEquals(NextWordProtocol.Outcome.VIOLATION, NextWordProtocol.classify("le chat voit la maison dans"));
    }

    @Test
    public void testStatistics() {
        assertEquals(2 * Math.pow(0.5, 10), NextWordProtocol.signTestP(10, 0), 1e-12);
        assertEquals(1.0, NextWordProtocol.signTestP(5, 5), 1e-12);
        assertEquals(1.0, NextWordProtocol.signTestP(0, 0), 1e-12);
        double[] ci = NextWordProtocol.wilson(18, 30);
        assertEquals(0.423, ci[0], 2e-3);
        assertEquals(0.754, ci[1], 2e-3);
    }
}
