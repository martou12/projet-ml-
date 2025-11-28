from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


def evaluate_classifier(model, X_train, y_train, X_test, y_test, name="model"):
    """
    Affiche accuracy, ROC-AUC, matrice de confusion et classification report
    pour un modèle de classification binaire.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"=== {name} ===")
    print("Accuracy:", round(acc, 3))
    print("ROC-AUC :", round(auc, 3))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))

    return acc, auc
