package RN.transformer;

import java.util.List;

public class Batch {
    private List<String> data;
    private List<String> target;

    public Batch(List<String> data, List<String> target) {
        this.data = data;
        this.target = target;
    }

    public List<String> getData() {
        return data;
    }

    public List<String> getTarget() {
        return target;
    }
}
