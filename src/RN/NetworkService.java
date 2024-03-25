package RN;

import org.apache.log4j.Logger;

/**
 * @author Eric Marchand
 * 
 */
public class NetworkService {
	
	
	private static Logger logger = Logger.getLogger(NetworkService.class);
	
	public static void saveNetwork(Long idAdapter, Long idSource){
		
//		DBConnection con = TestNetwork.getNetconnection();
//		Network network = TestNetwork.getNetwork(idAdapter, idSource);
//		if(network == null)
//			network = TestNetwork.addNewNetwork(idAdapter, idSource);
//		insertNodes(con, network);
		
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
