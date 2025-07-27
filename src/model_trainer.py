from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

def train_and_evaluate():
    X = df.drop(columns=["species"])
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

#from src.model import train_and_evaluate

if __name__ == "__main__":
    # df = sys.argv[1]
    model, acc = train_and_evaluate()
    print(f"Model trained successfully.\n Accuracy: {acc:.4f}")

