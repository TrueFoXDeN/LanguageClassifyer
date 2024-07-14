import torch
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from app import TextClassifier  # Importiere das Modell aus der Trainingdatei


def predict(model, vectorizer, text, label_encoder):
    model.eval()
    text_vectorized = vectorizer.transform([text]).toarray()
    text_tensor = torch.tensor(text_vectorized, dtype=torch.float32).cuda()
    with torch.no_grad():
        output = model(text_tensor)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()


def main():
    # Modell, Vektorisierer und Label-Encoder laden
    print("Lade das Modell...")
    input_dim = 10000  # Dies sollte übereinstimmen mit max_features in CountVectorizer
    hidden_dim = 128
    output_dim = 2  # Englisch und Deutsch

    model = TextClassifier(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load('language_classifier_model.pth'))
    model = model.cuda()

    print("Lade den Vektorisierer und den Label-Encoder...")
    vectorizer = joblib.load('vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    # Beispielhafte Nutzung des Modells zur Vorhersage
    example_text = "Geben Sie hier Ihren Text ein"
    prediction = predict(model, vectorizer, example_text, label_encoder)
    language = label_encoder.inverse_transform([prediction])[0]
    print(f"Der eingegebene Text ist in {language}.")


if __name__ == "__main__":
    main()
