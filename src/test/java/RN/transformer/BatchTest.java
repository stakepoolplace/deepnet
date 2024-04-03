package RN.transformer;

import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

public class BatchTest {
    
    @Test
    public void testDataAndTargetAssignment() {
        List<String> data = Arrays.asList("data1", "data2");
        List<String> target = Arrays.asList("target1", "target2");
        Batch batch = new Batch(data, target);
        
        Assert.assertEquals("Data should match input list", data, batch.getData());
        Assert.assertEquals("Target should match input list", target, batch.getTarget());
    }
}
