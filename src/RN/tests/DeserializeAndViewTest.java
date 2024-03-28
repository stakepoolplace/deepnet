package RN.tests;

import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.HashSet;
import java.util.Set;

import RN.Network;
import RN.NetworkService;

public class DeserializeAndViewTest {

    public static void main(String[] args) {
        // Remplacez le chemin vers votre fichier .ser
        String filePath = "models/deeper-net-model-20240328021700.ser";

        try (FileInputStream fileIn = new FileInputStream(filePath);
             ObjectInputStream in = new ObjectInputStream(fileIn)) {

            // Désérialise l'objet depuis le fichier
        	Network object = (Network) in.readObject();

            // Affiche des informations sur l'objet désérialisé
            // Cette partie dépend du type d'objet et de ce que vous souhaitez visualiser
            System.out.println("Objet désérialisé (toString): " + object.toString());
//            System.out.println(object.getString());
            
            Set<Integer> seenObjects = new HashSet<>();
            System.out.println("Objet désérialisé: ");
            NetworkService.printObjectDetails(object, 0, seenObjects);


        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Erreur lors de la lecture du fichier .ser: " + e.getMessage());
        }
                
        
    }
    
 
    
}
