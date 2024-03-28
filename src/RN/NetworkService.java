package RN;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Set;

import org.apache.log4j.Logger;

/**
 * @author Eric Marchand
 * 
 */
public class NetworkService {

	private static Logger logger = Logger.getLogger(NetworkService.class);

    public static void saveNetwork(Object reseauNeurone, String filename) throws Exception {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(reseauNeurone);
        } catch (FileNotFoundException e) {
            throw new Exception("File not found", e);
        } catch (IOException e) {
            throw new Exception("Error initializing stream", e);
        }
    }


	
    public static Network loadNetwork(String filename) throws Exception {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            return (Network) ois.readObject();
        } catch (FileNotFoundException e) {
            throw new Exception("File not found", e);
        } catch (IOException e) {
            throw new Exception("Error initializing stream", e);
        } catch (ClassNotFoundException e) {
            throw new Exception("Class not found", e);
        }
    }
    
    public static String printObjectDetails(Object object, int level, Set<Integer> seenObjects) throws IllegalAccessException {
        StringBuilder sb = new StringBuilder();
        
        if (object == null) {
            return "null";
        }
        
        int objectId = System.identityHashCode(object);
        if (seenObjects.contains(objectId)) {
            return "\"[Circular Reference]\"";
        }

        Class<?> objClass = object.getClass();
        seenObjects.add(objectId);
        String indent = repeat(" ", level * 2);

        if (objClass.isArray()) {
            sb.append("[\n");
            int length = Array.getLength(object);
            for (int i = 0; i < length; i++) {
                sb.append(indent).append(printObjectDetails(Array.get(object, i), level + 1, seenObjects));
                if (i < length - 1) sb.append(",\n");
            }
            sb.append("\n").append(indent).append("]");
        } else if (objClass.isPrimitive() || isWrapperType(objClass)) {
            sb.append(object.toString());
        } else if (objClass.equals(String.class)) {
            sb.append("\"").append(object).append("\"");
        } else {
            boolean isFirstField = true;
            sb.append("{\n");
            Field[] fields = objClass.getDeclaredFields();
            for (Field field : fields) {
                if (!Modifier.isStatic(field.getModifiers()) && !field.isSynthetic() && !estUnAttributFonctionnel(field)) {
                    if (!isFirstField) {
                        sb.append(",\n");
                    } else {
                        isFirstField = false;
                    }
                    field.setAccessible(true);
                    sb.append(indent).append("  \"").append(field.getName()).append("\": ")
                      .append(printObjectDetails(field.get(object), level + 1, seenObjects));
                }
            }
            if (!isFirstField) {
                sb.append("\n");
            }
            sb.append(indent).append("}");
        }
        if (level == 0) {
            sb.append("\n"); // Ajouter une nouvelle ligne à la fin pour le niveau racine
        }
        
        return sb.toString();
    }



    private static boolean estUnAttributFonctionnel(Field field) {
    	
    	if(field.getName().equals("function")) {
    		return true;
    	}
        // Exemple simple : vérifier si le type du champ est une interface fonctionnelle du package java.util.function
        return java.util.function.Function.class.isAssignableFrom(field.getType());
        // Ajoutez d'autres conditions selon les types spécifiques que vous souhaitez ignorer
    }

    private static boolean isWrapperType(Class<?> clazz) {
        return clazz.equals(Boolean.class) || clazz.equals(Integer.class) ||
               clazz.equals(Character.class) || clazz.equals(Byte.class) ||
               clazz.equals(Short.class) || clazz.equals(Double.class) ||
               clazz.equals(Long.class) || clazz.equals(Float.class);
    }
    
    public static String repeat(String str, int count) {
        StringBuilder sb = new StringBuilder();
        
        for (int i = 0; i < count; i++) {
            sb.append(str);
        }
        
        return sb.toString();
    }
	

//	private static void insertNodes(final DBConnection connection, Network network){
//		
//        try {
//
//            if (logger.isDebugEnabled()) {
//                logger.debug("INSERT Node : ");
//            }
//
//            for(Layer layer : network.getLayers()){
//            
//            	for(Area area : layer.getAreas()){
//            		
//            		for(Node node : area.getNodes()){
//            
//            			for(Link link : node.getInputs()){
//            			connection.executeUpdate(DBSqlCommand.PS_NN_INSERT_NODE, 
//            					network.getAdapterId(),
//            					network.getNetworkId(),
//            					layer.getLayerId(), 
//            					area.getAreaId(), 
//            					node.getNodeId(), 
//            					node.getFunction().name(), 
//            					link.getLinkId(), 
//            					link.getWeight());
//            			}
//            			
//            			if(node.getBiasInput() != null){
//                			connection.executeUpdate(DBSqlCommand.PS_NN_INSERT_NODE, 
//                					network.getAdapterId(),
//                					network.getNetworkId(),
//                					layer.getLayerId(), 
//                					area.getAreaId(), 
//                					node.getNodeId(), 
//                					node.getFunction().name(), 
//                					node.getBiasInput().getLinkId(), 
//                					node.getBiasInput().getWeight());
//            			}
//            
//            		}
//            	}
//            
//            }
//
////            recordSource.setId(connection.getGeneratedId());
//            connection.commit();
//            
////            if(connection.getGeneratedId()!=null)
////                recordSource.setId(connection.getGeneratedId());
//
//        } catch (final SQLException e) {
//            connection.rollback();
//            logger.error(e.getMessage());
//        }
//		
//	}

//	public static Network loadNetwork(Long idAdapter, Long idNetwork){
//		DBConnection con = TestNetwork.getNetconnection();
//		return fetchNetwork(idAdapter, idNetwork, con);
//	}

//	private static Network fetchNetwork(Long idAdapter, Long idNetwork, DBConnection con){
//		
//		Network network = TestNetwork.getNetwork(idAdapter, idNetwork);
//		DBResultSet rs = null;
//		
//		try{
//				rs = con.executeQuery(
//		                DBSqlCommand.PS_NN_LST_NODES, 
//		                idAdapter,
//		                idNetwork);
//				
//				Layer layer = null;
//				Area area = null;
//				Node node = null;
//				Link link = null;
//		        while(rs.next()){
//		        
//		        	network.setAdapterId(rs.getInt("id_adapter"));
//		        	network.setSourceId(rs.getInt("id_source"));
//		            layer = network.getLayer(rs.getInt("id_layer"));
//		            area = layer.getArea(rs.getInt("id_area"));
//		            node = area.getNode(rs.getInt("id_node"));
//		            node.setFunction(EActivation.getEnum(rs.getString("activation")));
//		            if(rs.getInt("id_link") == -1){
//		            	link = node.getBiasInput();
//		            	link.setWeight(rs.getDouble("weight"));
//		            }else{
//		            	link = node.getInput(rs.getInt("id_link"));
//		            	link.setWeight(rs.getDouble("weight"));
//		            }
//		        }
//		        
//		        rs.close();
//		        rs = null;
//
//        		
//    } catch (final SQLException sqle) {
//        logger.error(sqle);
//    } finally {
//        if (rs != null) {
//            rs.close();
//            rs = null;
//        }
//    }
//		
//		return network;
//        
//	}
//	

}
