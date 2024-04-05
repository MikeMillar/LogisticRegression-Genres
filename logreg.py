import numpy as np
import utils

class LogisticRegression:
    def __init__(self, learning_rate=0.01, penalty=1.0, epsilon=1e-6, max_iterations=10000) -> None:
        """
        Initalizes the Logistic Regression model.

        Args:
            learning_rate (float): The learning rate or step size used in the gradient. Default 0.01
            penalty (float): The penalty to apply to weights to ensure they don't grow too large. Default 0.0 (no penalty)
            epsilon (float): The early termination condition value to check against. Default 1e-9
            max_iterations (int): The max gradient ascent iterations before guaranteed termination. Default 10000

        Returns:
            Nothing.
        """
        self.learning_rate = learning_rate
        self.default_learning_rate = learning_rate
        self.penalty = penalty
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.classes = None
        self.weights = None

    def exp(self, X):
        """
        This method takes a value X and returns the natural
        number 'e' raised to X. This function is used in the
        calculation of P(Y = y_k | X) as provided by EQ 27
        in Mitchell. The denominator has been removed from that
        equation, in favor of normalizing the values so that
        EQ 28 does not need to be used to compute class C
        independently.

        Args:
            X ([[x]]): arraylike of numbers.

        Return:
            ([[x]]): The result of e^X in the same shape of X,
            where each row is normalized.
        """
        max_per_row = np.max(X, axis=1, keepdims=True)
        z = np.exp(X - max_per_row)
        return z / np.sum(z, axis=1, keepdims=True)
    
    def encode(self, Y):
        """
        Takes an arraylike of targets and converts it into
        the huffman encoding where each row has exactly one
        value of 1, indicating that the target is of the 
        class indicated by the column.

        Args:
            Y ([y]): Arraylike of target classes

        Returns:
            ([[y]]): A matrix of the encoded Y array
        """
        return (Y[:, None] == self.classes).astype(int)
    
    def add_intercept(self, X):
        """
        Adds a column of ones to the X matrix.

        Args:
            X ([[x]]): Matrix to add column to.

        Returns:
            ([[x]]): Original matrix with a column of ones
            added to the front.
        """
        if len(X.shape) == 1:
            intercept = np.ones(X.shape[0])
            return np.array([intercept, X]).T
        else:
            intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def random_weights(self, num_features):
        """
        Creates a randomized set of weights from a normal distribution of shape
        (num_classes, num_features).

        Args:
            num_features (int): Total number of features, and weights required.

        Returns: 
            ([[x]]): A matrix of shape (num_classes, num_features) with randomized values.
        """
        return np.zeros((len(self.classes), num_features))
        # return np.random.normal(0, 1, (len(self.classes), num_features))

    def squared_error(self, errors):
        """
        Given a matrix of values which is assumed to be the difference of the the
        actual value and the predicted value, computes the mean squared error
        from that difference.

        Args:
            errors ([[x]]): Matrix of differences

        Returns:
            (float): Mean squared error value
        """
        return np.sum(np.square(errors)) / errors.shape[0]

    def fit(self, X, Y) -> None:
        """
        Trains the logistic regression model by computing a set of
        weight coefficients which can be used to classify new data
        instances based on teh classes in the training set.

        Args:
            X (df): Dataframe of data to train on. This data should not
                    contain the labels of the data.
            Y (np.array): An ordered array of the labels of the data in X.

        Returns:
            Nothing.
        """
        # Save unique classes
        self.classes, self.class_mapping = np.unique(Y, return_inverse=True)
        # Convert dataframe to a numpy array with added intercept
        X = self.add_intercept(np.array(X))
        # Encode the target values using huffman coding
        Y_encoded = self.encode(Y)
        # Initalize weight matrix with random values
        W = self.random_weights(X.shape[1])
        # Set up previous iteration error
        last_error = float('inf')
        min_error = float('inf')
        
        # Gradient Ascent - loop until early termination or max iterations
        for i in range(self.max_iterations):
            # Overflow occured, exit program
            if np.sum(np.isnan(W)) > 0:
                print('NaN detected on iteration:', i)
                exit()
            # calculate P(Y = y_k | X) -> calling costs for lack of better name
            probs = self.exp(np.dot(X, W.T))
            # costs = self.exp(np.dot(X, W.T))
            # Calculate error
            error = self.squared_error(Y_encoded - probs)
            # error = self.cross_entropy(Y_encoded, probs)
            error_diff = abs(error - last_error)
            min_error = min(error_diff, min_error)
            # print('Error:', error, '\tDiff:', error_diff)
            # Check for early termination
            if error_diff < self.epsilon:
                # print(f'Early termination on iter={i} with error={error}')
                break
            # Update last error
            last_error = error
            # Calculate pentalties -> lambda * W
            penalties = self.penalty * W
            # Calculate gradient -> G = (erorrs.T - penalties) / total_instances
            gradient = np.dot(X.T, Y_encoded - probs).T - penalties
            # Calculate new W -> W = W + (eta) * (errors - penalties)
            W_new = W + (self.learning_rate * gradient)
            # Update old weights with new weights
            W = W_new

        # Set our trained weights
        self.weights = W
        # print('Smallest error diff:', min_error)

    def compute_probabilities(self, X):
        """
        Given a data matrix X and the trained weights W,
        computes the probability that any X_q is of class C.

        Args:
            X ([[x]]): Matrix of data

        Returns:
            (([x])): Matrix of probabilities of shape (n, c). 
            where n is the total rows of data in X and C is the
            total number of classes.
        """
        return self.exp(np.dot(X, self.weights.T))

    def predict(self, X) -> None:
        """
        Using the previously trained weights (W), computes the 
        normalized probabilities that each instance of X is
        of any class. Then returns the value of the class that
        has the highest probability for each instance of X.

        Args:
            X (df): Dataframe of data

        Returns:
            Y ([y]): Arraylike of predicted labels for each 
                     instance of X.
        """
        # Compute probabilities of each instance for each class
        probs = self.compute_probabilities(self.add_intercept(np.array(X)))
        # Determine the column index with the highest probability
        class_idx = np.argmax(probs, axis=1)
        # Return labels of highest prediction
        return self.classes[class_idx]