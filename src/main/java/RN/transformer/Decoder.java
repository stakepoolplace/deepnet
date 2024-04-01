package RN.transformer;

public class Decoder {
    private Attention selfAttention;
    private Attention encoderAttention;
    private FeedForward feedForward;
    
    public Decoder(int numHeads, int modelSize, int ffDim) {
        this.selfAttention = new Attention(numHeads, modelSize);
        this.encoderAttention = new Attention(numHeads, modelSize);
        this.feedForward = new FeedForward(modelSize, ffDim);
    }
    
    // Implement decode function here
}
