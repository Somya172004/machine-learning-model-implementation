# Display classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
plt.matshow(conf_matrix, cmap='Blues', fignum=1)
plt.title("Confusion Matrix")
plt.colorbar()
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.yticks([0, 1], ['Ham', 'Spam'])
plt.show()
