import torch
import joblib
import torch.nn.functional as F
from train import TextClassifier


def predict(model, vectorizer, text, label_encoder, device):
    model.eval()
    text_vectorized = vectorizer.transform([text]).toarray()
    text_tensor = torch.tensor(text_vectorized, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(text_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        return predicted.item(), confidence.item()


def main():
    while True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Modell, Vektorisierer und Label-Encoder laden

        input_dim = 10000  # Dies sollte übereinstimmen mit max_features in CountVectorizer
        hidden_dim = 128
        output_dim = 2  # Englisch und Deutsch

        model = TextClassifier(input_dim, hidden_dim, output_dim)
        model.load_state_dict(torch.load('language_classifier_model.pth'))
        model = model.to(device)

        vectorizer = joblib.load('vectorizer.pkl')
        label_encoder = joblib.load('label_encoder.pkl')

        print('Eingabe:')
        # Beispielhafte Nutzung des Modells zur Vorhersage
        example_text = input()
        prediction, confidence = predict(model, vectorizer, example_text, label_encoder, device)
        language = label_encoder.inverse_transform([prediction])[0]
        print(f"Der eingegebene Text ist in {language} mit einer Konfidenz von {confidence:.2f}.\n")


if __name__ == "__main__":
    main()
