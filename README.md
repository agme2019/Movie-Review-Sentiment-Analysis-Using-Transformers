# Movie-Review-Sentiment-Analysis-Using-Transformers
Created this project to apply my learnings on Transformers, used the IMDB dataset

The first model codes are available in the "Simple" [here](myLib/README.md) folder. The test accuracy is ~ **75 %**.
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

