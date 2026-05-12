# toxic_comment_detection_lstm.py
"""Toxic comment detection pipeline using TF-IDF and a neural network classifier."""

from __future__ import annotations

import random
import re
import warnings
import pickle
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from nltk.stem import PorterStemmer
import nltk

warnings.filterwarnings('ignore')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False


CURRENT_DIR = Path(__file__).parent

DATASET_PATH = CURRENT_DIR / "dummy_toxic_comment_dataset.csv"
MODEL_PATH = CURRENT_DIR / "toxic_comment_model.pkl"
VECTORIZER_PATH = CURRENT_DIR / "toxic_comment_vectorizer.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2
MAX_FEATURES = 2000
BATCH_SIZE = 16
EPOCHS = 100


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and validate the toxic comment dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    expected_cols = {"Comment", "Label"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {expected_cols}, got {set(df.columns)}"
        )

    df = df[["Comment", "Label"]].copy()
    df["Comment"] = df["Comment"].astype(str)
    df["Label"] = df["Label"].astype(str)
    df = shuffle(df, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


stemmer = PorterStemmer()
stop_words = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """Clean and preprocess text: lowercase, tokenize, remove stopwords, stem."""
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)


def preprocess_texts(texts: pd.Series) -> Tuple[List[str], List[int]]:
    """Preprocess a series of texts."""
    cleaned_texts: List[str] = []
    token_lengths: List[int] = []

    for text in texts.tolist():
        cleaned = clean_text(text)
        cleaned_texts.append(cleaned)
        token_lengths.append(len(cleaned.split()) if cleaned else 0)

    return cleaned_texts, token_lengths



def encode_labels(labels: pd.Series) -> np.ndarray:
    """Encode labels to numeric values."""
    label_map = {"Non-Toxic": 0, "Toxic": 1}
    encoded = labels.map(label_map)

    if encoded.isna().any():
        unknown = sorted(labels[encoded.isna()].unique().tolist())
        raise ValueError(f"Unknown labels found in dataset: {unknown}")

    return encoded.astype(int).to_numpy()



def build_vectorizer(texts: List[str], max_features: int = MAX_FEATURES) -> TfidfVectorizer:
    """Build TF-IDF vectorizer for text feature extraction."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=1,
        max_df=0.9,
        ngram_range=(1, 2),
        lowercase=True,
    )
    vectorizer.fit(texts)
    return vectorizer


def texts_to_features(vectorizer: TfidfVectorizer, texts: List[str]) -> np.ndarray:
    """Convert texts to TF-IDF feature vectors."""
    features = vectorizer.transform(texts)
    return features.toarray()



def build_model(input_dim: int) -> MLPClassifier:
    """Build and compile a neural network classifier."""
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=VALIDATION_SPLIT,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    return model



def plot_training_history(train_losses: List[float], val_losses: List[float], 
                          train_scores: List[float], val_scores: List[float]) -> None:
    """Plot training history (accuracy and loss curves)."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_scores, label="Train Accuracy", marker='o', linewidth=2)
    plt.plot(epochs, val_scores, label="Validation Accuracy", marker='s', linewidth=2)
    plt.title("Accuracy Graph", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURRENT_DIR / "accuracy_graph.png", dpi=100, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker='o', linewidth=2)
    plt.plot(epochs, val_losses, label="Validation Loss", marker='s', linewidth=2)
    plt.title("Loss Graph", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURRENT_DIR / "loss_graph.png", dpi=100, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Non-Toxic", "Toxic"]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=14,
                fontweight='bold',
            )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(CURRENT_DIR / "confusion_matrix.png", dpi=100, bbox_inches='tight')
    plt.show()


def plot_toxic_word_frequency(df: pd.DataFrame) -> None:
    """Plot toxic word frequency and/or word cloud."""
    toxic_text = " ".join(df.loc[df["Label"] == "Toxic", "Comment"].astype(str))
    toxic_clean = clean_text(toxic_text)

    if not toxic_clean.strip():
        print("No toxic words found to visualize.")
        return

    if WORDCLOUD_AVAILABLE:
        try:
            wordcloud = WordCloud(width=900, height=450, background_color="white").generate(toxic_clean)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title("Toxic Word Cloud", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(CURRENT_DIR / "wordcloud.png", dpi=100, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Error generating wordcloud: {e}")
            plot_toxic_word_frequency_fallback(toxic_clean)
    else:
        plot_toxic_word_frequency_fallback(toxic_clean)


def plot_toxic_word_frequency_fallback(toxic_clean: str) -> None:
    """Fallback: plot word frequency bar chart."""
    tokens = toxic_clean.split()
    freq = pd.Series(tokens).value_counts().head(15)
    plt.figure(figsize=(10, 5))
    freq.plot(kind="bar", color="coral")
    plt.title("Top 15 Toxic Word Frequency", fontsize=14, fontweight='bold')
    plt.xlabel("Word", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CURRENT_DIR / "word_frequency.png", dpi=100, bbox_inches='tight')
    plt.show()


def get_top_comment_tokens(comment: str, vectorizer: TfidfVectorizer, top_n: int = 5) -> list[tuple[str, float]]:
    """Return the top TF-IDF tokens from a comment."""
    cleaned = clean_text(comment)
    features = vectorizer.transform([cleaned]).toarray()[0]
    token_index = {index: token for token, index in vectorizer.vocabulary_.items()}
    token_values = [(token_index[idx], float(value)) for idx, value in enumerate(features) if value > 0]
    return sorted(token_values, key=lambda x: x[1], reverse=True)[:top_n]


def plot_class_word_clouds(df: pd.DataFrame) -> None:
    """Create side-by-side word clouds for toxic and non-toxic comments."""
    toxic_text = " ".join(df.loc[df["Label"] == "Toxic", "Comment"].astype(str))
    non_toxic_text = " ".join(df.loc[df["Label"] == "Non-Toxic", "Comment"].astype(str))

    toxic_clean = clean_text(toxic_text)
    non_toxic_clean = clean_text(non_toxic_text)

    if WORDCLOUD_AVAILABLE:
        wordcloud_toxic = WordCloud(width=450, height=450, background_color="white").generate(toxic_clean)
        wordcloud_non_toxic = WordCloud(width=450, height=450, background_color="white").generate(non_toxic_clean)

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(wordcloud_toxic, interpolation="bilinear")
        plt.title("Toxic Word Cloud")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(wordcloud_non_toxic, interpolation="bilinear")
        plt.title("Non-Toxic Word Cloud")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(CURRENT_DIR / "class_word_clouds.png", dpi=100, bbox_inches='tight')
        plt.show()
    else:
        print("WordCloud is not available, skipping class word clouds.")


def save_evaluation_report(report_text: str) -> None:
    """Save evaluation report to a text file."""
    report_path = CURRENT_DIR / "evaluation_report.txt"
    report_path.write_text(report_text, encoding="utf-8")


def predict_comment(
    comment: str,
    vectorizer: TfidfVectorizer,
    model: MLPClassifier,
) -> Tuple[str, float, np.ndarray]:
    """Predict toxicity of a comment and return class probabilities."""
    cleaned = clean_text(comment)
    features = vectorizer.transform([cleaned]).toarray()

    probabilities = model.predict_proba(features)[0]
    score = float(probabilities[1])

    prediction = model.predict(features)[0]
    label = "Toxic" if prediction == 1 else "Non-Toxic"

    return label, score, probabilities


class TrainingHistory:
    """Track training history for plotting."""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_scores = []
        self.val_scores = []


def train_model_with_history(
    model: MLPClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    history: TrainingHistory,
) -> MLPClassifier:
    """Train the model and track history manually."""
    
    model_warm = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=1,
        batch_size=BATCH_SIZE,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        warm_start=True,
        random_state=RANDOM_STATE,
        verbose=0,
    )

    for epoch in range(EPOCHS):
        model_warm.fit(X_train, y_train)

        train_pred = model_warm.predict(X_train)
        val_pred = model_warm.predict(X_val)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        train_loss = log_loss(y_train, model_warm.predict_proba(X_train))
        val_loss = log_loss(y_val, model_warm.predict_proba(X_val))

        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.train_scores.append(train_acc)
        history.val_scores.append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} - "
                  f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f} - "
                  f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        if epoch > 10 and len(history.val_losses) > 10:
            if val_loss > min(history.val_losses[-10:]) * 1.05:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    return model_warm



def main() -> None:
    """Main pipeline: load data, train model, evaluate, and predict."""
    set_seed()

    print("\n" + "="*70)
    print("TOXIC COMMENT DETECTION USING NLP AND MACHINE LEARNING")
    print("="*70 + "\n")

    print("TASK 1: DATASET CREATION & LOADING")
    print("-" * 70)
    df = load_dataset(DATASET_PATH)

    print(f"\nDataset loaded successfully!")
    print(f"Total records: {len(df)}")
    print("\nDataset Preview (first 10 records):")
    print(df.head(10).to_string())
    print("\nLabel Distribution:")
    label_counts = df["Label"].value_counts()
    print(label_counts.to_string())
    print(f"\nPercentage of Toxic comments: {(label_counts.get('Toxic', 0) / len(df) * 100):.2f}%")
    print(f"Percentage of Non-Toxic comments: {(label_counts.get('Non-Toxic', 0) / len(df) * 100):.2f}%")

    print("\n" + "="*70)
    print("TASK 2: NLP PREPROCESSING")
    print("="*70 + "\n")
    print("Preprocessing steps:")
    print("  1. Lowercasing")
    print("  2. Tokenization")
    print("  3. Stopword removal")
    print("  4. Stemming\n")

    cleaned_texts, token_lengths = preprocess_texts(df["Comment"])

    print("Sample Cleaned Comments:")
    print("-" * 70)
    for i, (original, cleaned) in enumerate(zip(df["Comment"].head(5), cleaned_texts[:5]), 1):
        print(f"\n{i}. Original: {original}")
        print(f"   Cleaned : {cleaned}")

    max_len = int(max(token_lengths)) if token_lengths else 1
    max_len = max(max_len, 1)

    print(f"\n\nPreprocessing Statistics:")
    print(f"  - Sequence length (max tokens): {max_len}")
    print(f"  - Average tokens per comment: {np.mean(token_lengths):.2f}")
    print(f"  - Min tokens in a comment: {int(min(token_lengths))}")
    print(f"  - Max tokens in a comment: {int(max(token_lengths))}")

    labels = encode_labels(df["Label"])

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        cleaned_texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    X_train_texts, X_val_texts, y_train, y_val = train_test_split(
        X_train_texts,
        y_train,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    print(f"\n\nFeature Extraction (TF-IDF Vectorization)...")
    vectorizer = build_vectorizer(X_train_texts, max_features=MAX_FEATURES)
    vocab_size = len(vectorizer.vocabulary_)

    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Max features: {MAX_FEATURES}")
    print(f"  - N-gram range: (1, 2) [unigrams and bigrams]")

    X_train = texts_to_features(vectorizer, X_train_texts)
    X_val = texts_to_features(vectorizer, X_val_texts)
    X_test = texts_to_features(vectorizer, X_test_texts)

    print(f"\nFeature Matrix Shapes:")
    print(f"  - Training: {X_train.shape}")
    print(f"  - Validation: {X_val.shape}")
    print(f"  - Testing: {X_test.shape}")

    print("\n" + "="*70)
    print("TASK 3: LSTM MODEL DEVELOPMENT (Neural Network Classifier)")
    print("="*70 + "\n")
    print("Building neural network classifier...")
    model = build_model(input_dim=X_train.shape[1])

    print("\nModel Architecture:")
    print(f"  - Input Layer: {X_train.shape[1]} features (TF-IDF vectors)")
    print(f"  - Hidden Layer 1: 128 neurons with ReLU activation")
    print(f"  - Hidden Layer 2: 64 neurons with ReLU activation")
    print(f"  - Output Layer: 1 neuron with Sigmoid activation (binary classification)")
    print(f"  - Optimizer: Adam")
    print(f"  - Loss Function: Binary Crossentropy")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Validation Split: {VALIDATION_SPLIT * 100}%\n")

    print("Training model...")
    history = TrainingHistory()
    model = train_model_with_history(model, X_train, y_train, X_val, y_val, history)

    print("\nTraining completed!")

    print("\n" + "="*70)
    print("TASK 4 & 5: MODEL EVALUATION AND VISUALIZATION")
    print("="*70 + "\n")
    
    print("Evaluating on test set...")
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_loss_score = log_loss(y_test, model.predict_proba(X_test))

    print(f"\nTest Set Performance:")
    print(f"  - Test Accuracy : {test_acc:.4f}")
    print(f"  - Test Loss     : {test_loss_score:.4f}")

    print("\n\nClassification Report:")
    print("-" * 70)
    print(classification_report(y_test, test_pred, target_names=["Non-Toxic", "Toxic"]))

    print("\nGenerating visualizations...")
    print("  1. Confusion Matrix...")
    plot_confusion_matrix(y_test, test_pred)

    print("  2. Accuracy and Loss Graphs...")
    plot_training_history(
        history.train_losses,
        history.val_losses,
        history.train_scores,
        history.val_scores,
    )

    print("  3. Toxic Word Visualization...")
    plot_toxic_word_frequency(df)
    print("  4. Class Word Clouds...")
    plot_class_word_clouds(df)

    report_text = (
        f"Test Accuracy : {test_acc:.4f}\n"
        f"Test Loss     : {test_loss_score:.4f}\n\n"
        f"Classification Report:\n{classification_report(y_test, test_pred, target_names=['Non-Toxic', 'Toxic'])}"
    )
    save_evaluation_report(report_text)
    print(f"  5. Evaluation report saved to evaluation_report.txt")

    pickle.dump(model, open(MODEL_PATH, 'wb'))
    pickle.dump(vectorizer, open(VECTORIZER_PATH, 'wb'))
    print(f"\nModels saved:")
    print(f"  - Model: {MODEL_PATH}")
    print(f"  - Vectorizer: {VECTORIZER_PATH}")

    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70 + "\n")

    examples = [
        "You are completely useless.",
        "I appreciate your effort.",
        "You should be ashamed of yourself.",
        "Thank you for your support!",
        "This is pure garbage and nonsense.",
    ]

    print("Pre-defined Examples:")
    print("-" * 70)
    for i, text in enumerate(examples, 1):
        label, score, probabilities = predict_comment(text, vectorizer, model)
        contribution = get_top_comment_tokens(text, vectorizer, top_n=5)
        confidence = max(probabilities)
        print(f"\n{i}. Input: \"{text}\"")
        print(f"   Output: {label} Comment")
        print(f"   P(Non-Toxic): {probabilities[0]:.4f}")
        print(f"   P(Toxic)    : {probabilities[1]:.4f}")
        print(f"   Confidence  : {confidence:.4f}")
        if contribution:
            contributions = ", ".join([f"{token}:{score:.4f}" for token, score in contribution])
            print(f"   Top tokens : {contributions}")

    print("\n" + "="*70)
    print("INTERACTIVE PREDICTION SYSTEM")
    print("="*70 + "\n")
    
    print("Enter comments to classify (type 'quit', 'exit', or 'q' to leave):\n")
    
    while True:
        user_comment = input(">>> Enter a comment: ").strip()
        normalized = user_comment.lower()
        if normalized in {'quit', 'exit', 'q'}:
            print("Exiting interactive prediction...")
            break
        if user_comment:
            label, score, probabilities = predict_comment(user_comment, vectorizer, model)
            confidence = max(probabilities)
            print(f"\n    Prediction: {label} Comment")
            print(f"    P(Non-Toxic): {probabilities[0]:.4f}")
            print(f"    P(Toxic)    : {probabilities[1]:.4f}")
            print(f"    Confidence  : {confidence:.4f}\n")
        else:
            print("    Please enter a valid comment.\n")

    print("\n" + "="*70)
    print("Thank you for using Toxic Comment Detection System!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
