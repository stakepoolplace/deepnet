package RN.transformer;


public class Encoder {
    private Attention attention;
    private FeedForward feedForward;
    
    public Encoder(int numHeads, int modelSize, int ffDim) {
        this.attention = new Attention(numHeads, modelSize);
        this.feedForward = new FeedForward(modelSize, ffDim);
    }
    
    // Implement encode function here
}
