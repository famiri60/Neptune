import os

# Specify the path to the main folder
main_folder_path = "./20_newsgroups"

# Initialize a counter for the number of files read
files_read_count = 0

# Iterate through subfolders
for subfolder_name in os.listdir(main_folder_path):
    subfolder_path = os.path.join(main_folder_path, subfolder_name)

    # Check if the subfolder is a directory
    if os.path.isdir(subfolder_path):
        print(f"Processing files in {subfolder_name}:")

        # Iterate through files in the subfolder
        for file_name in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file_name)

            # Check if the file is a regular file
            if os.path.isfile(file_path):
                # Read the contents of the file
                with open(file_path, 'r', errors='ignore') as file:
                    file_contents = file.read()
                                    # Increment the files_read_count
                files_read_count += 1

                # Do something with the file contents (e.g., print or process)
                #print(f"File {file_name} contents:\n{file_contents[:]}...\n")

# Print the total number of files read
print(f"Total number of files read: {files_read_count}")

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer  # Replace PorterStemmer with WordNetLemmatizer
from nltk import FreqDist
from nltk import download
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import re
import time
from collections import Counter

# Download NLTK resources
download('stopwords')
download('punkt')
download('wordnet')  # Download WordNet data for lemmatization
download('averaged_perceptron_tagger')

# Load the 20 Newsgroups dataset
#newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
newsgroups = fetch_20newsgroups(subset='all')
# Create a DataFrame from the dataset
df = pd.DataFrame({'text': newsgroups.data, 'target': newsgroups.target})
df['Subject'] = df['text'].apply(lambda text: re.search(r'Subject: (.+)', text).group(1) if re.search(r'Subject: (.+)', text) else '')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.3, random_state=42, stratify=df['target'])

from nltk import pos_tag

from nltk.corpus import wordnet

# Define the get_wordnet_pos function
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if the POS tag is not found

# Text preprocessing pipeline (tokenization, lemmatization, stop words removal)
stop_words = set(stopwords.words('english')+['would'])
lemmatizer = WordNetLemmatizer()

def terms_count(text):
    # Original terms
    original_words = set(word_tokenize(text))

    words = word_tokenize(text.lower())
    # Terms after text.lower()
    lower_case_words = set(words)

    words = [word for word in words if word.isalpha()]  # Filter out non-alphabetic characters
    # Terms after filtering out numbers
    alpha_words = set(words)

    words_pos = pos_tag(words)  # POS tagging
    lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos)) for word, pos in words_pos]
    # Terms after lemmatization
    lemmatized_terms = set(lemmatized_words)

    words = [word for word in lemmatized_words if word.lower() not in stop_words]  # Remove stop words
    # Terms after removing stopwords
    without_stopwords_terms = set(words)

    return original_words, lower_case_words, alpha_words, lemmatized_terms, without_stopwords_terms


def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha()]  # Filter out non-alphabetic characters
    words_pos = pos_tag(words)  # POS tagging
    lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos)) for word, pos in words_pos]
    words = [word for word in lemmatized_words if word.lower() not in stop_words]  # Remove stop words
    text = ' '.join(words)
    return text


# Create a list of preprocessed documents
preprocessed_documents = [preprocess_text(doc) for doc in X_train]
preprocessed_docs = [terms_count(doc) for doc in X_train]
#print(preprocessed_docs)


common_words_list= [get_common_words(t) for t in preprocessed_documents]
print('Number of preprocessed document:', len(preprocessed_documents))


# Calculate total number of terms at each step
total_original_terms = set()
total_lower_case_terms = set()
total_alpha_terms = set()
total_lemmatized_terms = set()
total_without_stopwords_terms = set()

for terms_tuple in preprocessed_docs:
    total_original_terms.update(terms_tuple[0])
    total_lower_case_terms.update(terms_tuple[1])
    total_alpha_terms.update(terms_tuple[2])
    total_lemmatized_terms.update(terms_tuple[3])
    total_without_stopwords_terms.update(terms_tuple[4])

# Display the results
print(f"Total number of original terms: {len(total_original_terms)}")
print(f"Total number of lower case terms: {len(total_lower_case_terms)}")
print(f"Total number of alpha terms: {len(total_alpha_terms)}")
print(f"Total number of lemmatized terms: {len(total_lemmatized_terms)}")
print(f"Total number of terms after removing stopwords: {len(total_without_stopwords_terms)}")

# Create a CountVectorizer
count_vectorizer = CountVectorizer()

# Fit and transform the documents to get the term frequency matrix
tf_matrix = count_vectorizer.fit_transform(preprocessed_documents)

# Get the feature names (terms)
feature_names = count_vectorizer.get_feature_names_out()

# Map features to original words
original_words = []
for feature in feature_names:
    original_word = ' '.join([lemmatizer.lemmatize(word) for word in feature.split() if word.isalpha()])
    original_words.append(original_word)


# Fit and transform the documents to get the term frequency matrix
#tf_matrix = count_vectorizer.fit_transform(preprocessed_documents)


# Create a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
# Fit and transform the documents to get the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_documents)

# Calculate TF-IDF scores for each term in each document
tf_scores = pd.DataFrame(data=tf_matrix.toarray(), columns=original_words)

# Calculate TF-IDF scores for each term in each document
tfidf_scores = pd.DataFrame(data=tfidf_matrix.toarray(), columns=original_words)

# Calculate the entropy for each document
#entropy_values = -np.sum(normalize(tfidf_scores, norm='l1', axis=1) * np.log2(normalize(tfidf_scores, norm='l1', axis=1) + 1e-10), axis=1)

# Create a DataFrame to store the sum of TF-IDF scores for each term and each class
term_frequency_df = pd.DataFrame(data=tf_matrix.toarray(), columns=original_words)
term_frequency_tfidf = pd.DataFrame(data=tfidf_matrix.toarray(), columns=original_words)
term_frequency_df['target'] = y_train.reset_index(drop=True)
term_frequency_tfidf['target'] = y_train.reset_index(drop=True)

# Calculate the sum of TF-IDF scores for each term and each class
term_frequency_by_class = term_frequency_df.groupby('target').sum()
term_frequency_tfidf_class = term_frequency_tfidf.groupby('target').sum()

# Display the top N terms for each class
top_n_terms = 10

# Lists to store data for the DataFrame
class_names = []
class_names_tfidf = []
most_frequent_words = []
most_frequent_words_tfidf = []

# Dictionary to store data
class_term_frequency_dict = {}
class_term_frequency_dict_tfidf = {}

# Iterate over each class for term frequency
for class_index in range(len(term_frequency_df['target'].unique())):
    class_terms = term_frequency_df[term_frequency_df['target'] == class_index].iloc[:, :-1].sum().nlargest(top_n_terms)
    class_name = newsgroups.target_names[class_index]

    # Add class name and corresponding term-frequency tuple to the dictionary
    class_term_frequency_dict[class_name] = [(term, int(term_frequency)) for term, term_frequency in zip(class_terms.index, class_terms)]

# Iterate over each class for TF-IDF
for class_index_tfidf in range(len(term_frequency_tfidf_class)):
    class_terms_tfidf = term_frequency_tfidf_class.iloc[class_index_tfidf].nlargest(top_n_terms)
    class_name_tfidf = newsgroups.target_names[class_index_tfidf]

    # Add class name and corresponding term-frequency tuple to the dictionary
    class_term_frequency_dict_tfidf[class_name_tfidf] = [(term, int(term_frequency)) for term, term_frequency in zip(class_terms_tfidf.index, class_terms_tfidf)]

# Display the dictionary
print(class_term_frequency_dict)
print(class_term_frequency_dict_tfidf)

# Convert values into strings
string_dict = {key: ', '.join([f'{word} ({count})' for word, count in value]) for key, value in class_term_frequency_dict.items()}
string_dict_tfidf = {key: ', '.join([f'{word} ({count})' for word, count in value]) for key, value in class_term_frequency_dict_tfidf.items()}

# Create a DataFrame from the string_dict
df_freq = pd.DataFrame.from_dict(string_dict, orient='index', columns=['Most Frequent Words'])
df_freq_tfidf = pd.DataFrame.from_dict(string_dict_tfidf, orient='index', columns=['Most Frequent Informative Words'])

# Reset the index to get "Class" as a column
df_freq = df_freq.reset_index()
df_freq_tfidf = df_freq_tfidf.reset_index()
df_freq.columns = ["Class", "Most Frequent Words (TF Score)"]
df_freq_tfidf.columns = ["Class", "Most Frequent Informative words (TF_IDF Score)"]
df_freq = df_freq.merge(df_freq_tfidf, on='Class')
# Now, 'df_freq' contains the data as you described

df_freq.head()

# Save the DataFrame to an Excel file
#df_freq.to_excel('frequent_words_ten.xlsx', index=False)

from matplotlib import pyplot as plt
import seaborn as sns

# Group numbers to group names mapping
group_names = [newsgroups.target_names[i] for i in range(len(newsgroups.target_names))]

# Replace group numbers with group names in the DataFrame
df['target_names'] = df['target'].map(dict(enumerate(group_names)))

# Plotting
group_counts = df['target_names'].value_counts()

# Bar plot with seaborn
sns.barplot(y=group_counts.index, x=group_counts.values, palette=sns.palettes.mpl_palette('Dark2'))

# Remove unnecessary spines
plt.gca().spines[['top', 'right']].set_visible(False)

# Add x-axis and y-axis labels
plt.xlabel('Count')
plt.ylabel('Target Class')

# Show the plot
plt.show()

import matplotlib.pyplot as plt

# Data for the bar plot
labels = ['Original', 'Lower Case', 'Alpha_terms', 'Lemmatized', 'After Removing Stopwords']
counts = [len(total_original_terms), len(total_lower_case_terms), len(total_alpha_terms), len(total_lemmatized_terms), len(total_without_stopwords_terms)]

# Create a bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, counts, color=['blue', 'orange', 'purple', 'green', 'red'])
plt.title('Total Number of Terms at Each Preprocessing Step')
plt.xlabel('Preprocessing Steps')
plt.ylabel('Total Number of Terms')

# Add text labels on top of each bar
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.05, str(count), fontsize=10)

plt.show()

import plotly.graph_objects as go
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ... (your code for data loading and preprocessing)
X_train_p = X_train.apply(preprocess_text)
X_test_p = X_test.apply(preprocess_text)

# Function to calculate cosine similarity
def cosine_similarity_score(text, reference_text):
    vectorizer = TfidfVectorizer()
    text_vector = vectorizer.fit_transform([text, reference_text])
    similarity_score = cosine_similarity(text_vector[0], text_vector[1])[0, 0]
    return similarity_score

# Add a reference text for each class (you can use the average text in each class)
reference_texts = []
for class_index in range(len(newsgroups.target_names)):
    class_texts = X_train_p[y_train == class_index]
    average_text = ' '.join(class_texts)
    reference_texts.append(average_text)

# Create a graph
G = nx.Graph()

# Add nodes (newsgroup target names) to the graph
for class_index, class_name in enumerate(newsgroups.target_names):
    G.add_node(class_index, label=class_name)

# Add edges (similarity scores) to the graph
for i in range(len(newsgroups.target_names)):
    for j in range(i + 1, len(newsgroups.target_names)):
        similarity_score = cosine_similarity_score(reference_texts[i], reference_texts[j])
        G.add_edge(i, j, weight=similarity_score)

# Get node positions using a layout algorithm
pos = nx.spring_layout(G)  # You can use different layouts

# Extract node and edge positions
node_x = []
node_y = []
for key, value in pos.items():
    node_x.append(value[0])
    node_y.append(value[1])

edge_traces = []

# Find the maximum similarity score for each node
max_similarity_scores = {node: max(edge[2]['weight'] for edge in G.edges(node, data=True)) for node in G.nodes}

for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]

    # Determine if the edge has the highest similarity score for either node
    is_max_similarity = edge[2]['weight'] == max_similarity_scores[edge[0]] or edge[2]['weight'] == max_similarity_scores[edge[1]]

    # Create edge trace
    edge_trace = go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        line=dict(width=4 * edge[2]['weight'] if is_max_similarity else edge[2]['weight'], color='grey' if is_max_similarity else '#888'),
        hoverinfo='text',
        mode='lines'
    )

    # Add labels to edges
    edge_trace.text = f'Similarity: {edge[2]["weight"]:.4f}'

    edge_traces.append(edge_trace)

# Create node trace
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    text=[class_name for class_name in nx.get_node_attributes(G, 'label').values()],
    marker=dict(
        showscale=False,
        colorscale='YlGnBu',
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        )
    )
)

# Create text annotations for nodes
node_annotations = [dict(
    x=pos[key][0],
    y=pos[key][1] + 0.03,  # Adjust the y value to move the text above the node
    text=label,
    showarrow=False,
    xanchor='center',  # Center the text
    yanchor='bottom',  # Anchor at the bottom of the text
    font=dict(size=10)
) for key, label in nx.get_node_attributes(G, 'label').items()]

# Create figure
fig = go.Figure(data=edge_traces + [node_trace], layout=go.Layout(
    showlegend=False,
    hovermode='closest',
    margin=dict(b=0, l=0, r=0, t=0),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),  # Adjust the x-axis range
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    annotations=node_annotations  # Add node labels
))

# Show figure
fig.show()

import plotly.graph_objects as go
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ... (your code for data loading and preprocessing)
X_train_p = X_train.apply(preprocess_text)
X_test_p = X_test.apply(preprocess_text)

# Function to calculate cosine similarity
def cosine_similarity_score(text, reference_text):
    vectorizer = TfidfVectorizer()
    text_vector = vectorizer.fit_transform([text, reference_text])
    similarity_score = cosine_similarity(text_vector[0], text_vector[1])[0, 0]
    return similarity_score

# Add a reference text for each class (you can use the average text in each class)
reference_texts = []
for class_index in range(len(newsgroups.target_names)):
    class_texts = X_train_p[y_train == class_index]
    average_text = ' '.join(class_texts)
    reference_texts.append(average_text)

# Create a matrix to store similarity scores
num_classes = len(newsgroups.target_names)
similarity_matrix = np.zeros((num_classes, num_classes))

# Populate the similarity matrix
for i in range(num_classes):
    for j in range(i + 1, num_classes):
        similarity_score = cosine_similarity_score(reference_texts[i], reference_texts[j])
        similarity_matrix[i, j] = similarity_score
        similarity_matrix[j, i] = similarity_score

# Create heatmap
heatmap = go.Figure(data=go.Heatmap(z=similarity_matrix, x=newsgroups.target_names, y=newsgroups.target_names, colorscale='Viridis'))

# Customize the layout
heatmap.update_layout(
    xaxis=dict(title='Newsgroup Target Names', tickangle=-90),
    yaxis=dict(title='Newsgroup Target Names'),
    title='Cosine Similarity Matrix between Newsgroup Target Names',
    width=900,  # Adjust the width of the heatmap to make it square
    height=900  # Adjust the height of the heatmap to make it square
)

# Show the heatmap
heatmap.show()

len(X_train)

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


# Get the TF matrix
tf_vectorizer = CountVectorizer(preprocessor=preprocess_text)
tf_matrix = tf_vectorizer.fit_transform(X_train)

# Calculate TF scores for each term in each document
tf_scores = pd.DataFrame(data=tf_matrix.toarray(), columns=tf_vectorizer.get_feature_names_out())

# Sort terms by TF
sorted_tf_df = tf_scores.sum().sort_values(ascending=False)

# Get the number of terms
num_terms = list(range(1, len(sorted_tf_df) + 1))

plt.figure(figsize=(10, 6))
plt.bar(num_terms, sorted_tf_df.values)  # Use plt.bar to create a bar chart
plt.xscale('log')
plt.yscale('log')
plt.title("TF vs Number of Terms")
plt.xlabel("Number of Terms")
plt.ylabel("TF Value")
plt.show()



# Create a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
# Fit and transform the documents to get the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_documents)

# Get the IDF values
idf_values = tfidf_vectorizer.idf_

# Create a DataFrame for IDF values
idf_df = pd.DataFrame(data={"Term": feature_names, "IDF": idf_values})

# Sort terms by IDF
sorted_idf_df = idf_df.sort_values(by="IDF", ascending=False)

# Get the number of terms
num_terms = list(range(1, len(sorted_idf_df) + 1))

plt.figure(figsize=(10, 6))
plt.bar(num_terms, sorted_idf_df["IDF"])  # Use plt.bar to create a bar chart
plt.xscale('log')  # Set the y-axis to a logarithmic scale
plt.title("IDF vs Number of Terms")
plt.xlabel("Number of Terms")
plt.ylabel("IDF")
plt.show()

# Create a CountVectorizer
count_vectorizer = CountVectorizer()

# Fit and transform the documents to get the term frequency matrix
tf_matrix = count_vectorizer.fit_transform(preprocessed_documents)

# Get the feature names (terms)
feature_names = count_vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a Pandas DataFrame
tfidf_df = pd.DataFrame(data=tfidf_matrix.toarray(), columns=feature_names)
# Calculate the sum of TF-IDF values for each term
tfidf_sum = tfidf_df.sum()

# Sort terms by TF-IDF
sorted_tfidf_df = tfidf_sum.sort_values(ascending=False)

# Get the number of terms
num_terms = list(range(1, len(sorted_tfidf_df) + 1))

plt.figure(figsize=(10, 6))
plt.bar(num_terms, sorted_tfidf_df.values)  # Use plt.bar to create a bar chart
plt.yscale('log')  # Set the y-axis to a logarithmic scale
#plt.xscale('log')
#plt.yscale('log')
plt.title("TF-IDF vs Number of Terms")
plt.xlabel("Number of Terms")
plt.ylabel("TF-IDF Value")
plt.show()

import numpy as np

def entropy(probabilities):
    # Calculate entropy for a list of probabilities
    return -np.sum(probabilities * np.log2(probabilities))

# Calculate entropy for each term
term_entropy = []
for term in feature_names:
    term_tf = tf_scores[term].sum()
    term_idf = idf_df[idf_df["Term"] == term]["IDF"].values[0]
    term_tfidf = tfidf_df[term].sum()
    term_probabilities = [term_tf, term_idf, term_tfidf]
    term_probabilities = term_probabilities / np.sum(term_probabilities)  # Normalize
    term_entropy.append(entropy(term_probabilities))

# Create a DataFrame for term entropy
entropy_df = pd.DataFrame(data={"Term": feature_names, "Entropy": term_entropy})
# Sort terms by Entropy
sorted_entropy_df = entropy_df.sort_values(by='Entropy', ascending=False)

# Get the number of terms
num_terms = list(range(1, len(sorted_entropy_df) + 1))

plt.figure(figsize=(10, 6))
plt.bar(num_terms, sorted_entropy_df['Entropy'])
plt.xscale('log')
plt.title("Entropy vs Number of Terms")
plt.xlabel("Number of Terms")
plt.ylabel("Entropy Value")
plt.show()

# Store the preprocessed text
X_train_preprocessed = X_train.apply(preprocess_text)
X_test_preprocessed = X_test.apply(preprocess_text)

# Create pipelines with TF-IDF vectorizer and different classifiers
models = {
    'Naive Bayes': make_pipeline(TfidfVectorizer(), MultinomialNB()),
    'Decision Tree': make_pipeline(TfidfVectorizer(), DecisionTreeClassifier()),
    'SVM': make_pipeline(TfidfVectorizer(), SVC()),
    'Random Forest': make_pipeline(TfidfVectorizer(), RandomForestClassifier())
}

# Find the best-performing model based on accuracy
best_model_name = None
best_model_accuracy = 0.0

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining and evaluating {model_name} model:")

    # Measure the execution time
    start_time = time.time()

    # Train the model using preprocessed text
    model.fit(X_train_preprocessed, y_train)

    # Make predictions on the test set using preprocessed text
    y_pred = model.predict(X_test_preprocessed)

    # Measure the prediction time
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    if accuracy > best_model_accuracy:
      best_model_accuracy = accuracy
      best_model_name = model_name

    # Print the results
    print(f"Accuracy: {accuracy}")
    print("\nClassification Report:")
    print(classification_rep)
    print("\nConfusion Matrix:")
    print(conf_matrix)

# Now, best_model_name contains the name of the best model, and you can use X_test_preprocessed for further steps.

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Example hyperparameter grid for GridSearchCV
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf'],
}

svm_pipeline = make_pipeline(TfidfVectorizer(), SVC())
grid_search = GridSearchCV(svm_pipeline, param_grid, cv=5)
grid_search.fit(X_train_preprocessed, y_train)

# Print the best parameters
print("Best parameters:", grid_search.best_params_)

# Make predictions using the best model
y_pred = grid_search.predict(X_test_preprocessed)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


# Print the results
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_rep)
print("\nConfusion Matrix:")
print(conf_matrix)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# Debugging lines
print("X_train.shape:", X_train_preprocessed.shape)

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.Softmax(dim=1)  # Softmax layer for classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        #x = self.softmax(x)  # Apply softmax for classification
        return x

# Create a pipeline with TF-IDF vectorizer and neural network
model_nn_pipeline = make_pipeline(
    TfidfVectorizer(),
    SimpleNN(input_size=1, hidden_size=128, output_size=len(newsgroups.target_names))
)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(model_nn_pipeline.named_steps['tfidfvectorizer'].fit_transform(X_train_preprocessed).toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.int64)

# Modify the input size of the neural network model
input_size = X_train_tensor.shape[1]
model_nn_pipeline.named_steps['simplenn'].fc1 = nn.Linear(input_size, 128)

X_test_tensor = torch.tensor(model_nn_pipeline.named_steps['tfidfvectorizer'].transform(X_test_preprocessed).toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.int64)

# Create model, loss function, and optimizer
model_nn = model_nn_pipeline.named_steps['simplenn']
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_nn.parameters(), lr=0.001)

# Training the neural network
num_epochs = 10
batch_size = 64

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Record the start time
start_time = time.time()
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model_nn(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
# Record the end time of training
training_time = time.time() - start_time
print(f"Training Time: {training_time} seconds")

# Record the start time of evaluation
start_time = time.time()
# Testing the neural network
with torch.no_grad():
    outputs = model_nn(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy_nn = accuracy_score(y_test_tensor, predicted)
    classification_rep_nn = classification_report(y_test_tensor, predicted)
    conf_matrix_nn = confusion_matrix(y_test_tensor, predicted)
# Record the end time of evaluation
evaluation_time = time.time() - start_time
print(f"Evaluation Time: {evaluation_time} seconds")

print(f"Neural Network Accuracy: {accuracy_nn}")
print("\nClassification Report:")
print(classification_rep_nn)
print("\nConfusion Matrix:")
print(conf_matrix_nn)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Convert data to a suitable format
#X_train = X_train_prepro.astype(str)
#X_test = X_test.astype(str)

# Create TF-IDF vectorizer with custom preprocessor
vectorizer = TfidfVectorizer()

# Fit and transform on training data
X_train_tfidf = vectorizer.fit_transform(X_train_preprocessed).toarray()
X_test_tfidf = vectorizer.transform(X_test_preprocessed).toarray()

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.int64)
X_test_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.int64)

# Define hyperparameter search space
param_grid = {
    'lr': [0.0001, 0.001],
    'hidden_size': [64, 128, 256],
    'batch_size': [32, 64],
}

# Generate all combinations of hyperparameters
hyperparameter_combinations = list(ParameterGrid(param_grid))

best_accuracy = 0.0
best_hyperparameters = None

for hyperparameters in hyperparameter_combinations:
    start_time = time.time()  # Record start time for the experiment
    # Modify the input size of the neural network model
    input_size = X_train_tfidf.shape[1]
    output_size = 20

    # Create model, loss function, and optimizer with hyperparameters
    model_nn = SimpleNN(input_size, hyperparameters['hidden_size'], output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_nn.parameters(), lr=hyperparameters['lr'])

    # Training the neural network
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)

    for epoch in range(10):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model_nn(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Testing the neural network
    with torch.no_grad():
        outputs = model_nn(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy_nn = accuracy_score(y_test_tensor, predicted)

    end_time = time.time()  # Record end time for the experiment
    elapsed_time = end_time - start_time  # Calculate elapsed time

    # Print results for each hyperparameter set
    print(f"Hyperparameters: {hyperparameters}")
    print(f"Neural Network Accuracy: {accuracy_nn}\n")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds\n")

    # Track the best hyperparameters
    if accuracy_nn > best_accuracy:
        best_accuracy = accuracy_nn
        best_hyperparameters = hyperparameters

# Print the best hyperparameters and corresponding accuracy
print("Best Hyperparameters:")
print(best_hyperparameters)
print(f"Best Accuracy: {best_accuracy}")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Convert labels to PyTorch tensors
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#X_train_preprocessed = X_train.apply(preprocess_text)
#X_test_preprocessed = X_test_incorrect.apply(preprocess_text)

# Tokenize training data
train_encodings = tokenizer(X_train_preprocessed.tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')

# Tokenize testing data
test_encodings = tokenizer(X_test_preprocessed.tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], y_train_tensor)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], y_test_tensor)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(np.unique(y_train)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 10  # Increase the number of epochs
early_stopping_patience = 1  # Number of epochs to wait for improvement

# Initialize early stopping variables
best_loss = float('inf')
no_improvement_counter = 0

# Record the start time
start_time = time.time()

all_preds = []
all_labels = []


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)  # Make sure to include labels
        loss = outputs.loss

        if loss is not None:  # Check if loss is provided
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(test_loader, desc='Validation'):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if loss is not None:
                val_loss += loss.item()
                logits = outputs.logits  # Compute logits
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(test_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')

    # Check for early stopping
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        no_improvement_counter = 0
    else:
        no_improvement_counter += 1
        if no_improvement_counter == early_stopping_patience:
            print(f'Early stopping at epoch {epoch + 1} as there is no improvement in validation loss.')
            break

# Record the end time of training
elapsed_time = time.time() - start_time
print(f"Training Time: {elapsed_time} seconds")


accuracy = accuracy_score(all_labels, all_preds)
classification_rep = classification_report(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f"BERT Model Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_rep)
print("\nConfusion Matrix:")
print(conf_matrix)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

# Define a new model for stacking
class StackingModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(StackingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 20)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Debugging lines
print("X_train.shape:", X_train_preprocessed.shape)


# Convert data to PyTorch tensors
X_train_tensor_nn = torch.tensor(model_nn_pipeline.named_steps['tfidfvectorizer'].fit_transform(X_train_preprocessed).toarray(), dtype=torch.float32)
X_test_tensor_nn = torch.tensor(model_nn_pipeline.named_steps['tfidfvectorizer'].transform(X_test_preprocessed).toarray(), dtype=torch.float32)

with torch.no_grad():
    train_preds_nn = model_nn(X_train_tensor)
    test_preds_nn = model_nn(X_test_tensor)

# Convert labels to PyTorch tensors for BERT
y_train_tensor_bert = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor_bert = torch.tensor(y_test.values, dtype=torch.long)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize training data
train_encodings = tokenizer(X_train_preprocessed.tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')

# Tokenize testing data
test_encodings = tokenizer(X_test_preprocessed.tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')
train_dataset_bert = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], y_train_tensor_bert)
test_dataset_bert = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], y_test_tensor_bert)

# Get predictions from BERT
model.eval()
all_preds_bert = []
with torch.no_grad():
    for input_ids, attention_mask, labels in tqdm(test_loader, desc='Evaluating BERT'):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        all_preds_bert.extend(preds.cpu().numpy())

# Combine predictions from Neural Network and BERT
# Ensure the dimensions match before concatenating
num_samples_train = min(train_preds_nn.shape[0], len(all_preds_bert))
num_samples_test = min(test_preds_nn.shape[0], len(all_preds_bert))

X_train_stacked = torch.cat((train_preds_nn[:num_samples_train], torch.tensor(all_preds_bert[:num_samples_train]).view(-1, 1)), dim=1)
X_test_stacked = torch.cat((test_preds_nn[:num_samples_test], torch.tensor(all_preds_bert[:num_samples_test]).view(-1, 1)), dim=1)

# Convert labels to PyTorch tensors for stacking
y_train_tensor_stacking = torch.tensor(y_train.values[:num_samples_train], dtype=torch.long)

# Create the stacking model
num_classes = len(newsgroups.target_names)  # Update with the actual number of classes
stacking_model = StackingModel(input_size=X_train_stacked.shape[1], num_classes=num_classes)
criterion_stacking = nn.CrossEntropyLoss()
optimizer_stacking = optim.Adam(stacking_model.parameters(), lr=0.001)

# Training the stacking model
num_epochs_stacking = 10
batch_size_stacking = 8

train_dataset_stacking = TensorDataset(X_train_stacked, y_train_tensor_stacking)
train_loader_stacking = DataLoader(train_dataset_stacking, batch_size=batch_size_stacking, shuffle=True)
for epoch in range(num_epochs_stacking):
    for batch_X, batch_y in train_loader_stacking:
        optimizer_stacking.zero_grad()
        outputs_stacking = stacking_model(batch_X)
        loss_stacking = criterion_stacking(outputs_stacking, batch_y)
        loss_stacking.backward()
        optimizer_stacking.step()

# Testing the stacking model
with torch.no_grad():
    outputs_stacking = stacking_model(X_test_stacked)
    _, predicted_stacking = torch.max(outputs_stacking, 1)
    accuracy_stacking = accuracy_score(y_test_tensor, predicted_stacking)
    classification_rep_stacking = classification_report(y_test_tensor, predicted_stacking)
    conf_matrix_stacking = confusion_matrix(y_test_tensor, predicted_stacking)

print(f"Stacking Model Accuracy: {accuracy_stacking}")
print("\nClassification Report:")
print(classification_rep_stacking)
print("\nConfusion Matrix:")
print(conf_matrix_stacking)



from transformers import AdamW, RobertaForSequenceClassification, RobertaTokenizer

# Load pre-trained RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# Convert labels to PyTorch tensors
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Tokenize training data
train_encodings = tokenizer(X_train_preprocessed.tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')

# Tokenize testing data
test_encodings = tokenizer(X_test_preprocessed.tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')

# Create PyTorch datasets
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], y_train_tensor)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], y_test_tensor)

# Create PyTorch data loaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained RoBERTa model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(np.unique(y_train)))

# Move model to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 10
early_stopping_patience = 1  # Set desired patience

# Initialize early stopping variables
best_loss = float('inf')
no_improvement_counter = 0

# Record the start time
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        # Move tensors to the GPU
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        # Backward pass
        loss.backward()
        # Update parameters
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(test_loader, desc='Validation'):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if loss is not None:
                val_loss += loss.item()
                logits = outputs.logits  # Compute logits
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(test_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')

    # Check for early stopping
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        no_improvement_counter = 0
    else:
        no_improvement_counter += 1
        if no_improvement_counter == early_stopping_patience:
            print(f'Early stopping at epoch {epoch + 1} as there is no improvement in validation loss.')
            break

# Record the end time of training
elapsed_time = time.time() - start_time
print(f"Training Time: {elapsed_time} seconds")

accuracy = accuracy_score(all_labels, all_preds)
classification_rep = classification_report(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f"RoBERT Model Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_rep)
print("\nConfusion Matrix:")
print(conf_matrix)

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


# Load pre-trained DistilBert model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Convert labels to PyTorch tensors
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Tokenize training data
train_encodings = tokenizer(X_train_preprocessed.tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')

# Tokenize testing data
test_encodings = tokenizer(X_test_preprocessed.tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], y_train_tensor)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], y_test_tensor)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=20)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)


num_epochs = 10
early_stopping_patience = 1

# Initialize early stopping variables
best_loss = float('inf')
no_improvement_counter = 0

# Record the start time
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(test_loader, desc='Validation'):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if loss is not None:
                val_loss += loss.item()
                logits = outputs.logits  # Compute logits
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())


    avg_val_loss = val_loss / len(test_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')

    # Check for early stopping
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        no_improvement_counter = 0
    else:
        no_improvement_counter += 1
        if no_improvement_counter == early_stopping_patience:
            print(f'Early stopping at epoch {epoch + 1} as there is no improvement in validation loss.')
            break

# Record the end time of training
elapsed_time = time.time() - start_time
print(f"Training Time: {elapsed_time} seconds")


accuracy = accuracy_score(all_labels, all_preds)
classification_rep = classification_report(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f"DistilBert Model Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_rep)
print("\nConfusion Matrix:")
print(conf_matrix)

from sklearn.ensemble import VotingClassifier
models = {
    'SVM': SVC(),
    'Naive Bayes': MultinomialNB()
    }

# Create a Voting Classifier
voting_model = VotingClassifier([(model_name, make_pipeline(TfidfVectorizer(), model)) for model_name, model in models.items()], voting='hard')

start_time = time.time()
# Train the ensemble model
voting_model.fit(X_train_preprocessed, y_train)

# Make predictions on the test set
y_pred_ensemble = voting_model.predict(X_test_preprocessed)

# Evaluate the ensemble model
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
classification_rep_ensemble = classification_report(y_test, y_pred_ensemble)
conf_matrix_ensemble = confusion_matrix(y_test, y_pred_ensemble)

end_time = time.time()  # Record end time for the experiment
elapsed_time = end_time - start_time  # Calculate elapsed time
# Print the results for the ensemble model
print("\nEnsemble Model (Voting Classifier) Results:")
print(f"Accuracy: {accuracy_ensemble}")
print("\nClassification Report:")
print(classification_rep_ensemble)
print("\nConfusion Matrix:")
print(conf_matrix_ensemble)
print(f"Elapsed Time: {elapsed_time:.2f} seconds\n")

from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import ParameterGrid
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Tokenize the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(X_train_preprocessed.tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')
test_encodings = tokenizer(X_test_preprocessed.tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')

# Create custom datasets
train_dataset = CustomDataset(train_encodings, y_train.values)
test_dataset = CustomDataset(test_encodings, y_test.values)

# Define the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(np.unique(y_train)))

# Define the Trainer
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Define the hyperparameter grid
param_grid = {
    'learning_rate': [1e-5, 2e-5, 3e-5],
    'per_device_train_batch_size': [4, 16],
    'num_train_epochs': [4],  # Fixed at 4 epochs
}

# Iterate over hyperparameter combinations
best_accuracy = 0
best_params = None

for params in ParameterGrid(param_grid):
    training_args = TrainingArguments(
        output_dir="./",
        evaluation_strategy="epoch",
        logging_dir="./logs",
        save_strategy="epoch",
        **params  # Include hyperparameters in TrainingArguments
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    result = trainer.evaluate()

    # Update the best parameters if the accuracy is improved
    if result["eval_accuracy"] > best_accuracy:
        best_accuracy = result["eval_accuracy"]
        best_params = params

# Print the best parameters and accuracy
print("Best Parameters:", best_params)
print("Best Accuracy:", best_accuracy)
