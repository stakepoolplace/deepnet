package RN.transformer;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class PositionalEncodingTest {

    @Test
    public void testGetPositionalEncodingShape() {
        int dModel = 512;
        long sequenceLength = 20;
        PositionalEncoding pe = new PositionalEncoding(dModel);
        
        INDArray posEncoding = pe.getPositionalEncoding(sequenceLength);
        
        // Vérifie la forme du tableau
        assertArrayEquals("La forme de l'encodage positionnel doit être [sequenceLength, dModel]",
                          new long[]{sequenceLength, dModel}, 
                          posEncoding.shape());
    }

    @Test
    public void testGetPositionalEncodingValues() {
        int dModel = 8;  // Taille réduite pour faciliter les vérifications
        long sequenceLength = 5;
        PositionalEncoding pe = new PositionalEncoding(dModel);
        
        INDArray posEncoding = pe.getPositionalEncoding(sequenceLength);

        // Vérifie la première position
        for (int j = 0; j < dModel; j++) {
            if (j % 2 == 0) {
                // Indices pairs doivent être 0.0 (sin(0) = 0)
                assertEquals("Position 0, indice pair j=" + j + " doit être 0.0",
                             0.0, posEncoding.getDouble(0, j), 1e-6);
            } else {
                // Indices impairs doivent être 1.0 (cos(0) = 1)
                assertEquals("Position 0, indice impair j=" + j + " doit être 1.0",
                             1.0, posEncoding.getDouble(0, j), 1e-6);
            }
        }

        // Vérifie les autres positions
        for (int pos = 1; pos < sequenceLength; pos++) {
            for (int j = 0; j < dModel; j++) {
                double angle = pos / Math.pow(10000.0, (2.0 * j) / dModel);
                double expectedValue = (j % 2 == 0) ? Math.sin(angle) : Math.cos(angle);
                assertEquals("Valeur à la position " + pos + ", indice " + j + " ne correspond pas à l'encodage attendu",
                             expectedValue, posEncoding.getDouble(pos, j), 1e-6);
            }
        }
    }

    @Test
    public void testGetPositionalEncodingSinCosPattern() {
        int dModel = 512;
        long sequenceLength = 20;
        PositionalEncoding pe = new PositionalEncoding(dModel);
        
        INDArray posEncoding = pe.getPositionalEncoding(sequenceLength);

        // Vérifie que les indices pairs sont des sin et les impairs des cos
        for (int pos = 0; pos < sequenceLength; pos++) {
            for (int j = 0; j < dModel; j++) {
                double angle = pos / Math.pow(10000.0, (2.0 * j) / dModel);
                double expectedValue = (j % 2 == 0) ? Math.sin(angle) : Math.cos(angle);
                assertEquals("Le motif sin/cos ne correspond pas à la position " + pos + ", indice " + j,
                             expectedValue, posEncoding.getDouble(pos, j), 1e-6);
            }
        }
    }
}
