import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from dataset import IMDBProcessor, IMDBDataset
from model import SentimentClassifier 
import time

# --- CONFIG ---
BATCH_SIZE = 64
HIDDEN_SIZE = 128
LR = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print("1. Loading Data...")
    dataset = load_dataset("imdb")
    train_raw = dataset['train']
    test_raw = dataset['test']

    print("2. processing Vocab & Embeddings...")
    processor = IMDBProcessor(max_vocab_size=35000, max_seq_len=300)
    processor.build_vocab(train_raw['text'])
    
    embedding_matrix = processor.load_pretrained_embeddings(embedding_dim=200)

    print("3. Creating DataLoaders...")
    train_dataset = IMDBDataset(train_raw, processor)
    test_dataset = IMDBDataset(test_raw, processor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"4. Initializing Model on {DEVICE}...")
    model = SentimentClassifier(embedding_matrix, HIDDEN_SIZE, output_size=1)
    model = model.to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"--- Epoch {epoch+1} Finished. Avg Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}% | Time: {time.time()-start_time:.1f}s ---\n")
    return model, processor


def predict_sentiment(review, model, processor):
    processed_review = processor.process_text(review)
    processed_review = processed_review.unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        prediction = torch.sigmoid(model(processed_review).squeeze()) > 0.5 
        sentiment = "Positive" if prediction.item() else "Negative"
        return sentiment

if __name__ == "__main__":
    model, processor = train()
    torch.save(model.state_dict(), "\Applications\Developer\python\Ai\SequenceModels\RNN\sentiment_class\src\sentiment_classifier.pt")
    print("Model saved to 'sentiment_classifier.pt'")
    print("Enter review to predict sentiment (or 'exit' to quit):")
    while True:
        review = input("Review: ")
        if review.lower() == "exit":
            break

        sentiment = predict_sentiment(review, model, processor)
        print(f"Sentiment: {sentiment}")
