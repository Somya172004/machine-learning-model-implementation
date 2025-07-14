# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data
X_test_tfidf = vectorizer.transform(X_test)

# Check the transformed data (sparse matrix)
print(f"TF-IDF Matrix Shape: {X_train_tfidf.shape}")
