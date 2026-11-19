package RN.transformer;

/**
 * Early stopping sur une métrique à MINIMISER (typiquement la perte de validation,
 * continue et peu bruitée — pas un taux de réussite sur 30 exemples).
 *
 * Une amélioration ne compte que si elle dépasse minDelta (absolu). L'arrêt est
 * demandé après `patience` évaluations consécutives sans amélioration significative.
 */
public class EarlyStopping {
    private final int patience;
    private final double minDelta;
    private double best = Double.POSITIVE_INFINITY;
    private int bestStep = -1;
    private int sinceBest = 0;

    public EarlyStopping(int patience, double minDelta) {
        if (patience < 1 || minDelta < 0) {
            throw new IllegalArgumentException("patience >= 1 et minDelta >= 0 requis");
        }
        this.patience = patience;
        this.minDelta = minDelta;
    }

    /** Enregistre une mesure ; retourne true si c'est un nouveau meilleur (à sauvegarder). */
    public boolean update(double metric, int step) {
        if (Double.isNaN(metric) || Double.isInfinite(metric)) {
            sinceBest++;
            return false;
        }
        if (metric < best - minDelta) {
            best = metric;
            bestStep = step;
            sinceBest = 0;
            return true;
        }
        sinceBest++;
        return false;
    }

    public boolean shouldStop() {
        return sinceBest >= patience;
    }

    public double getBest() {
        return best;
    }

    public int getBestStep() {
        return bestStep;
    }
}
