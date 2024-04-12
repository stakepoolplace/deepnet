package RN.algoactivations;


public class TestSoftmax2 {
	
    public static double[][] softmax(double[][] x) {
    	
        int rows = x.length;
        int cols = x[0].length;
        double[][] exp_x = new double[rows][cols];
        double[] maxPerColumn = new double[cols];
        
        // Trouver le maximum de chaque colonne pour éviter l'explosion exponentielle
        for (int j = 0; j < cols; j++) {
            maxPerColumn[j] = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < rows; i++) {
                if (x[i][j] > maxPerColumn[j]) {
                    maxPerColumn[j] = x[i][j];
                }
            }
        }

        // Calcul de exp(x_ij - max(x_j))
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                exp_x[i][j] = Math.exp(x[i][j] - maxPerColumn[j]);
            }
        }

        // Calculer la somme de chaque colonne
        double[] sumPerColumn = new double[cols];
        for (int j = 0; j < cols; j++) {
            sumPerColumn[j] = 0.0;
            for (int i = 0; i < rows; i++) {
                sumPerColumn[j] += exp_x[i][j];
            }
        }

        // Diviser chaque élément exp_x[i][j] par la somme de sa colonne pour normaliser
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                exp_x[i][j] /= sumPerColumn[j];
            }
        }

        return exp_x;
    }

    public static void main(String[] args) {
        double[][] x = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
        double[][] result = softmax(x);

        // Affichage du résultat pour vérification
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                System.out.printf("%.4f ", result[i][j]);
            }
            System.out.println();
        }
    }
}

