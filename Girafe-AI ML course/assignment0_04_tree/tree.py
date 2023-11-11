import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    prob = np.mean(y, axis=0)
    ntrop = -1*np.sum(prob*np.log(prob+EPS))
    
    return ntrop
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    prob = np.mean(y, axis=0)
    gn = 1 - np.sum(prob**2)
    
    return gn
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    # YOUR CODE HERE
    # mean = np.mean(y)
    # var = 
    return np.var(y)

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    
    return np.mean(y-np.meadian(y))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None

        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        split_mask = X_subset[:, feature_index] < threshold

        # Apply the mask to the feature matrix and the target values
        X_left = X_subset[split_mask]
        X_right = X_subset[~split_mask]
        y_left = y_subset[split_mask]
        y_right = y_subset[~split_mask]
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        split_mask = X_subset[:, feature_index] < threshold

        # Apply the mask to the target values
        y_left = y_subset[split_mask]
        y_right = y_subset[~split_mask]
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        best_criterion_value = float('-inf')  # Инициализируем наилучшее значение критерия как минус бесконечность
        best_feature_index = 0
        best_threshold = 0

        for feature_index in range(X_subset.shape[1]):
            # Для каждого признака
            feature_values = X_subset[:, feature_index]
            unique_values = np.unique(feature_values)
            
            for threshold in unique_values:
                # Для каждого уникального значения признака
                y_left, y_right = self.make_split_only_y(feature_index, threshold, X_subset, y_subset)
                
                n_left = len(y_left)         # Количество объектов в левой подвыборке
                n_right = len(y_right)       # Количество объектов в правой подвыборке
                n_total = n_left + n_right   # Общее количество объектов

                    
                criterion_value = self.criterion(y_subset) - ((n_left / n_total) * self.criterion(y_left)  + (n_right / n_total) * self.criterion(y_right))

                # Если текущее значение критерия лучше, чем наилучшее значение, обновите наилучшее значение
                if criterion_value > best_criterion_value:
                    best_criterion_value = criterion_value
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold
    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        if self.depth == self.max_depth or len(y_subset) < self.min_samples_split:
            # Создайте листовой узел с предсказанием
            mean = np.mean(y_subset, axis=0)
            return Node(0, 0, mean)
        
        feature_index, threshold = self.choose_best_split(X_subset, y_subset)

        # Создайте новый узел для текущего разделения.
        new_node = Node(feature_index, threshold)
        self.depth += 1
        
        # Разделите данные на две подвыборки по выбранному разделению.
        (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)

        # Рекурсивно постройте поддерево для левой и правой подвыборок.
        new_node.left_child = self.make_tree(X_left, y_left)
        new_node.right_child = self.make_tree(X_right, y_right)

        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)
            
        
        self.root = self.make_tree(X, y)
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        return np.array(one_hot_decode(self.predict_proba(X)))
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        y_predicted = []

        for sample in X:
            # Начните с корневого узла
            current_node = self.root
            while (current_node.left_child == None and current_node.right_child == None) == False:
                # Если значение признака меньше порога, переходим в левое поддерево
                if sample[current_node.feature_index] < current_node.value:

                    current_node = current_node.left_child
                else:
                    current_node = current_node.right_child

            # Добавляем предсказанный класс (листовой узел) к результату
            y_predicted.append(current_node.proba)

        return np.array(y_predicted)