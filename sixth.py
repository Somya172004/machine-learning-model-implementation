from sklearn.model_selection import cross_val_score

# Perform cross-validation on the Naive Bayes model
cv_scores = cross_val_score(nb_model, X_train_tfidf, y_train, cv=5, scoring='accuracy')

print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation accuracy: {cv_scores.mean():.2f}")
