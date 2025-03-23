# Movie-Review-Sentiment-Analysis-Using-Transformers
Created this project to apply my learnings on Transformers, used the IMDB dataset

The first model codes are available in the ["Simple"](https://github.com/agme2019/Movie-Review-Sentiment-Analysis-Using-Transformers/tree/main/Simple) folder. The test accuracy is ~ **75 %**.
Train and test data are 25k each. Each of the 25k data are equally divided between positive and negative reviews.
The model is first trained and validated on the train dataset and then tested on the "unseen" test data.

**Simple Transformers Model**

1. **Input Processing**:
   - Raw input text (example: "This movie was fantastic and I enjoyed every minute of it!")
   - Tokenization using the BERT tokenizer
   - Conversion to token IDs

2. **Embedding Layers**:
   - Word embedding layer (with embedding dimension of 256)
   - Positional encoding to provide sequence order information

3. **Transformer Encoder**:
   - 4 stacked transformer encoder layers
   - Each layer containing:
     - Multi-head self-attention (4 attention heads)
     - Feed-forward neural network (dimensionality of 1024)

4. **Classification Head**:
   - Linear layer that maps from the embedding dimension to 2 classes
   - Output probabilities for positive and negative sentiment
     
  <img width="546" alt="Screenshot 2025-03-18 at 22 24 40" src="https://github.com/user-attachments/assets/643a4758-cb4b-49a4-910d-c82434cef8d0" />

The diagram also highlights how the [CLS] token flows through the model and is ultimately used for classification, which is a common approach in transformer-based classification tasks.

**Experimental Results**

| Model ID | Test Accuracy | Architecture | Parameters | Notes |
|----------|---------------|--------------|------------|-------|
| TX-3 (Best) | 76.2% | 4 layers, 256 dim, 4 heads | 10.9M | Best overall performance |
| TX-5 | 75.5% | 4 layers, 512 dim, 8 heads | 28.23M | Larger model with more parameters |
| TX-2 | 75.6% | 4 layers, 128 dim, 4 heads | 4.7M | Fewer parameters |
| TX-1 | 75.4% | 2 layers, 128 dim, 4 heads | 4.3M | Smallest model |
| Back-translated | 75.4% | 4 layers, 256 dim, 4 heads | 10.9M | Used 85% more augmented data |

The best model appears to be TX-3 with a test accuracy of 0.762 (76.2%). This model uses 4 transformer layers, 256 embedding dimensions, and 4 attention heads, resulting in 10.9M parameters.

The back-translated data augmentation model achieved a test accuracy of 0.754 (75.4%), which is 0.8% lower than the best model. It uses the same architecture (4 layers, 256 embedding dimensions, 4 attention heads, 10.9M parameters) but incorporates 85% more training data through back-translation.

Interestingly, while the back-translation approach provides significantly more training data, it doesn't improve performance. The model shows a slightly lower validation accuracy (0.994 vs 0.997) and a lower test accuracy compared to TX-3. This suggests that simply increasing data volume through back-translation may not necessarily lead to better generalization on the IMDB sentiment classification task, and the original data may already contain sufficient information for this specific task.


