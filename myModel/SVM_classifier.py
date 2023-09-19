from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class ImageFeatureClassifier:
    def __init__(self):
        self.clf = SVC(kernel='linear', probability=True, random_state=42)

    def fit(self, X, y):
        """
        Train the SVM classifier on the provided image features.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Training data.
        - y: array-like, shape (n_samples,)
            Target values.
        """
        self.clf.fit(X, y)

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Data to classify.

        Returns:
        - array, shape (n_samples,)
            Predicted class label per sample.
        """
        return self.clf.predict(X)

    def predict_proba(self, X):
        """
        Return probability estimates for the test data.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Data for which to compute the predicted probabilities.

        Returns:
        - array, shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model.
        """
        return self.clf.predict_proba(X)
