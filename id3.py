import pandas as pd
import math
import pdb
from anytree import Node, RenderTree
from sklearn.tree import DecisionTreeClassifier, _tree, export_text
from sklearn.preprocessing import LabelEncoder


CLASSES = 2 # min of binary classification

def get_entropy(data, c_label):
    """
    Calculate the entropy for the given data and class label.
    Steps:
    1. Calculate the total number of samples in the data.
    2. For each unique class label, calculate the probability and its contribution to entropy.
    3. Sum the  entropies for each class label to get the final entropy.
    """
    # TOD: Implement entropy calculation logic here.
    total_num_of_samples = len(data[c_label])
    entropy = 0
    for val in data[c_label].unique():
        prob = data[c_label].value_counts()[val] / total_num_of_samples
        if prob > 0:
            entropy += -prob * math.log2(prob)
    return entropy

import math
import pandas as pd

def get_entropy(data, c_label):
    """
    Calculate the entropy for the given data and class label.
    """
    total_num_of_samples = len(data[c_label])  # Total number of samples
    entropy = 0
    
    for val in data[c_label].unique():
        prob = data[c_label].value_counts()[val] / total_num_of_samples  # Correct probability calculation
        if prob > 0:  # Avoid log2(0) error
            entropy += -prob * math.log2(prob)
    
    return entropy


    
def get_information_gain(data, feat_label, class_label):
    """
    Calculate the information gain for a given feature.
    Steps:
    1. Calculate the current entropy (without splitting) using get_entropy.
    2. For each unique value in the feature, calculate the entropy for that subset of the data.
    3. Subtract the weighted entropy for each subset from the current entropy to get the information gain.
    """
    # TOD: Implement information gain calculation logic here.
    initial_entropy = get_entropy(data, class_label)

    weighted_entropy = 0
    value_counts = data[feat_label].value_counts()  # Get counts of each feature value

    for val in data[feat_label].unique():
        prob = value_counts[val] / len(data)  # Probability of feature value
        subset_entropy = get_entropy(data[data[feat_label] == val], class_label)  # Entropy of subset
        weighted_entropy += prob * subset_entropy  # Weighted sum of entropy
    
    information_gain = initial_entropy - weighted_entropy
    return information_gain

def build_tree(data, features, c_label, T):
    """
    Recursively build a decision tree.
    Steps:
    1. If no data or features left, return None (base case).
    2. For each feature, calculate its information gain using get_information_gain.
    3. Pick the feature with the highest information gain as the splitting criterion.
    4. Recursively build the tree for each branch using the subset of data corresponding to each feature value.
    """
    # TOD: Implement tree construction logic here.
    # **Base Cases**
    if len(data) == 0 or len(features) == 0:
        return None
    
    # If all examples belong to the same class, return that class
    if len(data[c_label].unique()) == 1:
        return data[c_label].iloc[0]

    # **Step 2: Compute Information Gain for Each Feature**
    best_feature = None
    best_gain = 0
    
    for feature in features:
        gain = get_information_gain(data, feature, c_label)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    # **Step 3: If no Information Gain, return majority class**
    if best_gain == 0:
        return data[c_label].mode()[0]  # Most common class
    
    # **Step 4: Create a Decision Node**
    T[best_feature] = {}

    # **Step 5: Recursively Split Data**
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        subset = subset.drop(columns=[best_feature])  # Remove used feature
        
        # 🔥 **Fix: Pass `{}` as a new dictionary for each subtree**
        T[best_feature][value] = build_tree(subset, [f for f in features if f != best_feature], c_label, {})

    return T


def sklearn_decision_tree(dataframe, target_column):
    """
    Use Sklearn's decision tree to fit and print the tree structure.
    Steps:
    1. Encode categorical columns using encode_categorical.
    2. Separate features and target labels.
    3. Train a DecisionTreeClassifier on the data.
    4. Print the tree structure using export_text.
    """
    # TOD: Implement sklearn decision tree fitting and structure extraction logic here.
    encoded_df, encoders = encode_categorical(dataframe)

    X = encoded_df.drop(columns=[target_column])  # Features
    y = encoded_df[target_column]  # Target label

    model = DecisionTreeClassifier(criterion="entropy", random_state=42)  # Using ID3 (Entropy)
    model.fit(X, y)

    tree_rules = export_text(model, feature_names=list(X.columns))
    print("Decision Tree Structure:\n", tree_rules)

    return model, encoders
    pass

def encode_categorical(df):
    """
    Encode categorical features into numerical values using LabelEncoder.
    Steps:
    1. For each column, apply LabelEncoder to convert categorical values to integers.
    2. Return the encoded dataframe and the label encoders used.
    """
    # TOD: Implement categorical encoding logic here.
    encoders = {}  # Dictionary to store LabelEncoders
    encoded_df = df.copy()  # Copy DataFrame to avoid modifying original

    for col in df.columns:
        if df[col].dtype == 'object':  # Check if column is categorical
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(df[col])  # Convert to numerical labels
            encoders[col] = le  # Store encoder for later use
    
    return encoded_df, encoders




def convert_to_anytree(tree, parent_name="Root"):
    """ Converts your existing tree structure into an anytree format """
    root = Node(parent_name)
    
    def helper(subtree, parent):
        if not isinstance(subtree, dict):
            return
        for key, branches in subtree.items():
            child = Node(str(key), parent=parent)
            for branch_value, sub_branch in branches.items():
                branch_node = Node(f"{key}={branch_value}", parent=child)
                helper(sub_branch, branch_node)

    helper(tree, root)
    return root

def print_anytree(tree):
    """ Prints the decision tree in a structured way using anytree """


def fetch_and_clean():
    """
    Import and clean the mushroom dataset.
    Steps:
    1. Read the dataset using pandas.
    2. Drop any rows with missing values.
    3. Return the cleaned dataframe.
    """
    # TOD: Implement data fetching and cleaning logic here.
    df = pd.read_csv("mushroom.csv")
    df = df.dropna()  # Drop rows with missing values
    return df

if __name__ == "__main__":
    # df = fetch_and_clean('gain.csv')
    # info_gain_ice = get_information_gain(df, 'ice', 'school_cancelled?')
    # info_gain_above_freezing = get_information_gain(df, 'above_freezing', 'school_cancelled?')
    # print(f"We should take the higher information gain, which is ice: {max(info_gain_ice, info_gain_above_freezing)}")

    # Example use
    df = fetch_and_clean()
    c_label = 'class'
    CLASSES = len(df[c_label].unique())

    features = df.columns.values.tolist()
    features.remove(c_label)

    T = {}
    build_tree(df, features, c_label, T)

    anytree_root = convert_to_anytree(T)

    # YOUR TREE
    print_anytree(anytree_root)

    # SKLEARN TREE
    sklearn_decision_tree(dataframe=df, target_column=c_label)
