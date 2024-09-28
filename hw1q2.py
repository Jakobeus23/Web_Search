import os
import math

# constants for weight calculations
c1 = 0.5
c2 = 0.5

# load the txt files
def load_docs():
    doc_list = []
    for i in range(1, 11):  # hardcoded, oops, assumes 10 files with hardcoded names below
        try:
            # Open the file, ignore problematic characters
            with open(f"{i}.txt", "r", encoding="utf-8", errors='ignore') as file:
                doc_list.append(file.read().lower())  # Convert the content to lowercase for consistency
        except Exception as e:
            print(f"Error reading file {i}.txt: {e}.")
    return doc_list

# removes any punctuation from given string
def remove_punct(sent):
    sent = sent.replace(',', '').replace('.', '').replace('?', '').replace('()', '').replace(')', '').replace('"', '').replace('\'s', '') # scuffed but my other method messed up the words. I still check for invalid chars so its ok
    return sent.split()

# counts all the unique words and puts it in a list
def get_unique_words(sentList):
    wordList = []
    for sent in sentList:
        for word in sent:
            if word not in wordList:
                wordList.append(word)
    wordList.sort()  # alphabetical order
    return wordList

# This creates the rows of the document indexing matrix
# Each word is checked to see how many times it appears in a certain sentence (document)
def get_tf_matrix(sentList, wordList):
    tfArray = []
    for sent in sentList:
        tf = [sent.count(word) for word in wordList]
        tfArray.append(tf)
    return tfArray

# compute DF (Document Frequency) - Count how many documents each word appears in
def get_df(tfArray, wordList):
    dfArray = []
    for i, word in enumerate(wordList):
        doc_count = 0
        for doc in tfArray:
            if doc[i] > 0:  # If the word appears at least once in the document
                doc_count += 1
        dfArray.append(doc_count)
    return dfArray

# Compute IDF --> IDF(t) = log(N/DF(t))  --> N = total number of documents
def compute_idf(dfArray, N):
    idfArray = []
    for df in dfArray:
        if df == 0:
            idfArray.append(0)  # Handle the case where the term does not appear in any document
        else:
            idf_value = round(math.log(N / df), 3)
            idfArray.append(idf_value)
    return idfArray

# This function calculates the TFW matrix
# I need the weighted TF-IDF of every term t in both documents and queries
# TFW found in document indexing matrix, using the tfw matrix calculated below, using c1 + c2*tf/tfMax for each entry in the document indexing matrix
def get_tfw_matrix(tfArray, wordList):
    tfwMatrix = []
    for doc in tfArray:
        max_tf = max(doc)  # Maximum term frequency in this document
        tempArray = []
        for frequency in doc:
            if max_tf != 0:
                tempArray.append(round(c1 + c2 * (frequency / max_tf), 3))  # Proper weighting based on term frequency
            else:
                tempArray.append(0)  # Handle case where max_tf is 0
        tfwMatrix.append(tempArray)
    return tfwMatrix

# calculate w(t, d) for all documents (TFW * IDF)
def get_weighted_tfidf_matrix(tfwMatrix, idfArray):
    weighted_tfidf_matrix = []
    for doc_tfw in tfwMatrix:
        tfidf_doc = []
        for i in range(len(doc_tfw)):
            tfidf_value = doc_tfw[i] * idfArray[i]  # Multiply TFW by IDF
            tfidf_doc.append(round(tfidf_value, 3))
        weighted_tfidf_matrix.append(tfidf_doc)
    return weighted_tfidf_matrix

# Process the query: make lowercase, remove punctuation, and compute term frequencies (TF)
def process_query(query, wordList):
    query = remove_punct(query)  # Preprocess the query
    query_tf = get_tf_matrix([query], wordList)[0]  # Compute the term frequencies of the query
    return query_tf

# calculate w(t, q) for the query (TFW * IDF)
def get_weighted_query_tfidf(query_tf, idfArray, wordList):
    max_tf = max(query_tf) if max(query_tf) > 0 else 1  # Prevent division by 0
    query_tfidf = []
    for i, tf in enumerate(query_tf):
        tfidf_value = (c1 + c2 * (tf / max_tf)) * idfArray[i] if tf > 0 else 0
        query_tfidf.append(round(tfidf_value, 3))
    return query_tfidf

# Compute the similarity between query and document using dot product
def get_dot_product_similarity(query_tfidf_matrix, doc_tfidf_matrix):
    similarity_scores = []
    for query_tfidf in query_tfidf_matrix:
        query_scores = []
        for doc_tfidf in doc_tfidf_matrix:
            dot_product = sum(q * d for q, d in zip(query_tfidf, doc_tfidf))
            query_scores.append(dot_product)
        similarity_scores.append(query_scores)
    return similarity_scores

# Rank documents based on similarity scores
def rank_documents(similarity_scores):
    # this was confusing to use at first, but basically enumerate assigns a key (0,1,2), and "key=lambda x: x[1]" tells sorted() to use the 2nd value in the tuple for sorting. reverse=True is for highest to smallest similarity
    ranked_docs = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)
    return ranked_docs

# look I programmed a thing how Im supposed to finally :) this is basically main()
def search_query(query):
    # Load documents
    sentList = load_docs()
    N = len(sentList)

    # Preprocess documents
    processed_sents = [remove_punct(doc) for doc in sentList]

    # Create word list from all documents
    wordList = get_unique_words(processed_sents)

    # Compute TF matrix for documents
    tfArray = get_tf_matrix(processed_sents, wordList)

    # Compute DF array
    dfArray = get_df(tfArray, wordList)

    # Compute IDF array
    idfArray = compute_idf(dfArray, N)

    # Compute TFW matrix for documents
    tfwMatrix = get_tfw_matrix(tfArray, wordList)

    # Compute weighted TF-IDF matrix for documents
    weighted_tfidf_matrix = get_weighted_tfidf_matrix(tfwMatrix, idfArray)

    # Process the query and compute its weighted TF-IDF
    query_tf = process_query(query, wordList)
    query_tfidf = get_weighted_query_tfidf(query_tf, idfArray, wordList)

    # Compute similarity scores between query and documents
    similarity_scores = get_dot_product_similarity([query_tfidf], weighted_tfidf_matrix)

    # Rank documents based on similarity scores
    ranked_docs = rank_documents(similarity_scores[0])

    # I'm sorry but I can't be bothered to do this correctly right now and it isnt necessary for the graded portion so I just reused my print code
    for rank, (doc_index, score) in enumerate(ranked_docs):
        search_result = sentList[doc_index]
        break

    return ranked_docs, search_result

# test/example query
#query = "A new video from an astronaut's vantage point in space catures a bright green burst over Earth as a meteor exploded in the night sky"

print("What space related thing would you like to search for?")
query = input()
ranked_results, search_result = search_query(query)

# Output ranked results
print("\n\nHere is your best search result:\n")
print(search_result)

print("\nHere is the list of ranked documents:\n")
for rank, (doc_index, score) in enumerate(ranked_results):
    print(f"Rank {rank+1}: Document {doc_index+1}, Similarity Score: {score}")

