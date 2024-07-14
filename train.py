import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import joblib  # Für das Speichern des Vektorisierers


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()


class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print("Lade Trainingsdaten...")
    english_train = load_data('data/texts_en_nlp.txt')
    german_train = load_data('data/texts_de_nlp.txt')

    print("Lade Validierungsdaten...")
    english_val = load_data('data/validation_en.txt')
    german_val = load_data('data/validation_de.txt')

    print("Lade Testdaten...")
    english_test = load_data('data/test_en.txt')
    german_test = load_data('data/test_de.txt')

    # Labels hinzufügen
    print("Erstelle Labels für die Trainingsdaten...")
    train_texts = english_train + german_train
    train_labels = ['en'] * len(english_train) + ['de'] * len(german_train)

    print("Erstelle Labels für die Validierungsdaten...")
    val_texts = english_val + german_val
    val_labels = ['en'] * len(english_val) + ['de'] * len(german_val)

    print("Erstelle Labels für die Testdaten...")
    test_texts = english_test + german_test
    test_labels = ['en'] * len(english_test) + ['de'] * len(german_test)

    # Label Encoding
    print("Verschlüssele die Labels...")
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    val_labels = label_encoder.transform(val_labels)
    test_labels = label_encoder.transform(test_labels)

    # Tokenisierung und Vektorisierung
    print("Vektorisierung der Texte...")
    vectorizer = CountVectorizer(max_features=10000)
    train_texts_vectorized = vectorizer.fit_transform(train_texts).toarray()
    val_texts_vectorized = vectorizer.transform(val_texts).toarray()
    test_texts_vectorized = vectorizer.transform(test_texts).toarray()

    # Daten in PyTorch Dataset konvertieren
    print("Erstelle PyTorch Datasets...")
    train_dataset = TextDataset(train_texts_vectorized, train_labels)
    val_dataset = TextDataset(val_texts_vectorized, val_labels)
    test_dataset = TextDataset(test_texts_vectorized, test_labels)

    print("Erstelle PyTorch DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Modellarchitektur
    print("Definiere das Modell...")
    input_dim = train_texts_vectorized.shape[1]
    hidden_dim = 128
    output_dim = 2  # Englisch und Deutsch

    model = TextClassifier(input_dim, hidden_dim, output_dim)
    model = model.to(device)

    # Trainingsparameter
    print("Definiere die Trainingsparameter...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1

    for epoch in range(num_epochs):
        print(f'Starte Epoche {epoch + 1} von {num_epochs}...')
        model.train()
        running_loss = 0.0
        for texts, labels in train_loader:
            texts, labels = torch.tensor(texts, dtype=torch.float32).to(device), torch.tensor(labels,
                                                                                              dtype=torch.long).to(
                device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

        # Validation
        print("Validiere das Modell...")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = torch.tensor(texts, dtype=torch.float32).to(device), torch.tensor(labels,
                                                                                                  dtype=torch.long).to(
                    device)
                outputs = model(texts)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Validation Accuracy: {100 * correct / total:.2f} %')

    # Testen des Modells
    print("Teste das Modell...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = torch.tensor(texts, dtype=torch.float32).to(device), torch.tensor(labels,
                                                                                              dtype=torch.long).to(
                device)
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f} %')

    # Modell und Vektorisierer speichern
    print("Speichere das Modell und den Vektorisierer...")
    torch.save(model.state_dict(), 'language_classifier_model.pth')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Speichern abgeschlossen.")


if __name__ == "__main__":
    main()
