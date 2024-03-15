package RN.tests;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.Kernel;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.junit.Test;

import RN.linkage.GaborFilter;

/**
 * The test class for GaborFilter class
 */
public class GaborFilterTest {

   /**
    * A Gabor filter test
    *
    * @throws IOException - IOException
    */
   @Test
   public void testImage() throws IOException {

      Image imageLena = ImageIO.read(new File("./src/test/resources/lena.jpg"));
      
      GaborFilter gabor = null;
      Kernel kernel = null;
      
      File file = null;
      
      int orientationCount = 8;
      File fileFilter = null;
      BufferedImage bufferedImage = null;
      Graphics g = null;
      
      double[] orientations = new double[1];
      int angle;
      for(Integer orientationId=0; orientationId < orientationCount; orientationId++){
    	  
    	  angle = (int) ((360 / orientationCount) * orientationId);
    	  
    	  // Specifying the files
          file = new File("./src/test/resources/gaborred-lena-" + angle + ".jpg");
          fileFilter = new File("./src/test/resources/filter-" + angle + ".jpg");
          
    	  orientations[0] = (Math.PI / orientationCount) * orientationId;
    	  
	      // Creating buffered image from the given file. NOTE: It's crucial to build the data that way!
	      bufferedImage = new BufferedImage((int) imageLena.getWidth(null), imageLena.getHeight(null), BufferedImage.TYPE_INT_RGB);
	      g = bufferedImage.getGraphics();
	      g.drawImage(imageLena, 0, 0, null);
	      
	      
	      // Writing the filtered image to disk
	      gabor = new GaborFilter(16, orientations, 0, 0.5, 1, 7, 7);
	      ImageIO.write(gabor.filter(bufferedImage, null), "jpg", file);
	      
	      
    	  kernel = gabor.getKernel();
    	  
	      BufferedImage bufferedImageFilter = new BufferedImage(kernel.getWidth(), kernel.getHeight(), BufferedImage.TYPE_INT_RGB);
	      float data[] = kernel.getKernelData(null);
	      for(int x=0; x < kernel.getWidth(); x++){
	    	  for(int y=0; y < kernel.getWidth(); y++){
	    		  bufferedImageFilter.setRGB(x, y, Color.HSBtoRGB(1.2f, 1.0f, data[x*y + y]));
	    	  }
	      }
	      ImageIO.write(bufferedImageFilter, "jpg", fileFilter);
	      
      }
      
   }
   
   
   @Test
   public void testImage2() throws IOException {
	   // Specifying the files
	   File file = new File("./src/test/resources/gaborred-lena.jpg");
	   Image image = ImageIO.read(new File("./src/test/resources/lena.jpg"));
	
	   // Creating buffered image from the given file. NOTE: It's crucial to build the data that way!
	   BufferedImage bufferedImage = new BufferedImage(image.getWidth(null), image.getHeight(null), BufferedImage.TYPE_INT_RGB);
	   Graphics g = bufferedImage.getGraphics();
	   g.drawImage(image, 0, 0, null);
	
	   // Writing the filtered image to disk
	   GaborFilter gabor = new GaborFilter(16, new double[] {0, Math.PI/4, Math.PI}, 0, 0.5, 1, 3, 3);
	   ImageIO.write(gabor.filter(bufferedImage, null), "jpg", file);
   }
}