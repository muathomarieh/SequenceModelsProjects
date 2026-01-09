import torch
import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, output_size, num_layers=2, dropout=0.5):
        super(SentimentClassifier, self).__init__()
   
        num_embeddings, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        embeds = self.embedding(x)  
        _, (hn, cn) = self.lstm(embeds)
        last_hidden = hn[-1] 
        
        out = self.dropout(last_hidden)
        out = self.fc(out)
        
        return out