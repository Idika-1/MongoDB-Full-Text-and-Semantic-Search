# MongoDB Full-Text and Semantic Text Search

This project demonstrates how to implement **full-text search** and **semantic text search** on the Amazon Reviews dataset using **MongoDB** and **FAISS**.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation & Setup](#installation-setup)
  - [Install MongoDB & PyMongo](#install-mongodb-pymongo)
  - [Start MongoDB](#start-mongodb)
  - [MongoDB Setup with PyMongo](#mongodb-setup-with-pymongo)
- [Data Processing](#data-processing)
  - [Data Cleaning & Preprocessing](#data-cleaning-preprocessing)
  - [Convert Data to JSON](#convert-data-to-json)
  - [Insert Data into MongoDB](#insert-data-into-mongodb)
  - [Verify Data Insertion](#verify-data-insertion)
- [Search Implementation](#search-implementation)
  - [Full-Text Search](#full-text-search)
  - [Semantic Search](#semantic-search)
- [FAISS Integration](#faiss-integration)
  - [Setup FAISS](#setup-faiss)
  - [Optimized Semantic Search with FAISS](#optimized-semantic-search-with-faiss)
- [Conclusion](#conclusion)

## Project Overview

This project demonstrates how to:

1. Download and clean the Amazon Reviews dataset.
2. Insert the cleaned dataset into MongoDB.
3. Implement **full-text search** in MongoDB.
4. Set up **semantic search** using vector embeddings.
5. Integrate **FAISS** to enhance semantic search performance.

## Dataset

The project uses the **Amazon Reviews dataset**, which contains customer reviews, ratings, and metadata for various products sold on Amazon. This dataset will be cleaned and processed for search operations.

## Installation & Setup

### Install MongoDB & PyMongo

To get started, make sure **MongoDB** and **PyMongo** (Python driver for MongoDB) are installed.

1. **Install MongoDB**: Follow the [official MongoDB installation guide](https://docs.mongodb.com/manual/installation/).

2. **Install PyMongo**: Use the following pip command to install PyMongo:

   ```bash
   pip install pymongo

### Start MongoDB

Start MongoDB on your local machine:

```bash
mongod
```

By default, MongoDB runs on port `27017`.

### MongoDB Setup with PyMongo

To connect to MongoDB from Python:

```python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
# Sending a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

```

## Data Processing

### Data Cleaning & Preprocessing

The dataset will first need to be cleaned and preprocessed. This involves:

* Removing duplicates.
* Filling or removing missing values.
* Normalizing the text (lowercasing, removing special characters).

### Convert Data to JSON

After cleaning, convert the dataset into JSON format for MongoDB:

```python
# Assuming 'df' is your cleaned DataFrame
json_file = "Amazon_Reviews.json"
df.to_json(json_file, orient="records", lines=True)  #Line-delimited JSON data for MongoDB

print(f"CSV data successfully converted to JSON and saved as {json_file}")
```

### Insert Data into MongoDB

Once the data is in JSON format, insert it into MongoDB:

```python
#First, select the database and collection
db = client["my_database"]
collection = db["my_collection"]

#Loading the JSON data from file
with open(f"{json_file}", "r", encoding="utf-8") as file:
    #Reading each line in the file and loading it as a separate JSON document
    for line in file:
        try:
            data = json.loads(line)         #Deserialize each line of the file (i.e. converts each line into a Python dictionary)
            collection.insert_one(data)     #Inserts each line as a single document into the collection
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            continue   #Skipping the problematic line

print("Data successfully inserted")
```

### Verify Data Insertion

To check if the data was successfully inserted:

```python
print(collection.count_documents({}))  # Count number of documents in collection
print(collection.find_one())
```

## Search Implementation

### Full-Text Search

MongoDB supports **full-text search** through text indexes. Here's how to create an index on the review text field:

```python
collection.create_index([("reviewText", "text")])
```

Then, you can search for reviews containing a specific keyword:

```python
query = {"$text": {"$search": "great product"}}
results = collection.find(query)
for review in results:
    print(review)
```

### Semantic Search

To enable **semantic search**, you will need to convert the review text into vector embeddings (e.g., using a model like Sentence-BERT or other transformers). You can store these embeddings in MongoDB and later perform searches based on semantic similarity.

For generating embeddings using Sentence-BERT:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Example: Generate embeddings for reviews
embeddings = model.encode(df['reviewText'].tolist())
```

Store the embeddings along with the review data in MongoDB.

## FAISS Integration

### Setup FAISS

**FAISS** (Facebook AI Similarity Search) is used for fast similarity search. Install it using pip:

```bash
pip install faiss-cpu
```

For GPU support, use `faiss-gpu` instead of `faiss-cpu`.

### Optimized Semantic Search with FAISS

With FAISS, we can efficiently search for the most similar reviews by building an index from the review embeddings.

1. Convert the embeddings into a numpy array:

```python
import numpy as np
embeddings = np.array(embeddings)
```

2. Build an index using FAISS:

```python
import faiss

index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
index.add(embeddings)  # Add the embeddings to the index
```

3. To perform a search, generate an embedding for the query and search the FAISS index:

```python
query_embedding = model.encode("best product ever")
D, I = index.search(np.array([query_embedding]), k=5)  # Get top 5 nearest neighbors
```

FAISS will return the indices of the most similar reviews based on the query.

## Conclusion

This project shows how to set up **full-text search** and **semantic search** on Amazon review data using **MongoDB** and **FAISS**. By following the steps, you can preprocess large datasets, store them efficiently, and implement advanced search functionalities. FAISS enhances the performance of semantic search, making it faster and more scalable.

```
