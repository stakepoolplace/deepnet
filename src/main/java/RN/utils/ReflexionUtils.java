package RN.utils;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

/**
 * @author Eric Marchand
 *
 */
public class ReflexionUtils {
	
	public static <T>  T newClass(String classPath){
		return newClass(classPath, null, null);
	}
	
	public static <T>  T newClass(String classPath, Class[] paramClasses, Object... params){
		
		Class<T> classe = null;
		
		try {
			classe = (Class<T>) Class.forName(classPath);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		
		Constructor<T> constructeur = null;
		
		if(paramClasses != null && params != null){
			
			try {
				constructeur = classe.getConstructor(paramClasses);
			} catch (NoSuchMethodException | SecurityException e) {
				e.printStackTrace();
	//			try {
	//				constructeur = (Constructor<T>) classe.getSuperclass().getConstructor(paramClasses);
	//			} catch (NoSuchMethodException | SecurityException e1) {
	//				try {
	//					constructeur = (Constructor<T>) classe.getSuperclass().getSuperclass().getConstructor(paramClasses);
	//				} catch (NoSuchMethodException | SecurityException e2) {
	//					e2.printStackTrace();
	//				}
	//			}
			}
			
			try {
				return constructeur.newInstance(params);
			} catch (InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException e) {
				e.printStackTrace();
			}
			
		}else{
			
			try {
				constructeur = classe.getConstructor();
			} catch (NoSuchMethodException | SecurityException e) {
				e.printStackTrace();
			}
			
			try {
				return constructeur.newInstance();
			} catch (InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException e) {
				e.printStackTrace();
			}
			
		}
		
		return null;
		
	}

}
