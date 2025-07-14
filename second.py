# Preprocess the data
# Check for missing values
print(f"Missing values: {dataset.isnull().sum()}")

# Label encoding: 'ham' -> 0, 'spam' -> 1
dataset['Label'] = dataset['Label'].map({'ham': 0, 'spam': 1})

# Split the dataset into train and test sets
X = dataset['Message']
y = dataset['Label']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the split
print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
