import math

class Retrieve:
    def __init__(self, index, term_weighting):
        self.index = index
        #Term Weighting denotes the binary, tf or tfidf
        self.term_weighting = term_weighting
        #|D| = num_docs, total number of documents in the collection
        self.num_docs = self.compute_number_of_documents()
        #Get value of document freqency for each term
        self.dfw = self.get_dfw()
        #Get value of the inverse document frequency for each term
        self.idf = self.get_idf()
        #Get value of term frequencies for each document
        self.tf = self.get_tf()

        #Apply log normalization to term frequencies and get the value
        self.log_tf = self.get_log_tf(self.tf)
        #Apply maximum tf normalization to term frequencies and get the value
        self.ntf = self.get_ntf()
        #Get magnitude of each doc_vector
        self.doc_vector = self.get_doc_vector()

    def compute_number_of_documents(self):
        """Compute number of documents in the collection."""
        self.doc_ids = set()
        # Iterate over each term in the index
        for term in self.index:
            #Add doc_IDs for each term
            self.doc_ids.update(self.index[term].keys())
        #Return total no of documents
        return len(self.doc_ids)

    def get_dfw(self) -> dict:
        """Compute document frequency (DF) for each term."""
        dfw = {}
        #Iterate in the each term
        for w in self.index:
            # gets no of documents in the term
            dfw[w] = len(self.index[w]) 
        # Return the value of document frequencies to the dictionary
        return dfw

    def get_idf(self) -> dict:
        """Compute the Inverse Document Frequency (IDF) for each term."""
        idf = {}
        #Loop df through each term
        for w, df in self.dfw.items():
            if df > 0: #Reduce error by executing division by zero
                #Apply IDF formula
                idf[w] = math.log(self.num_docs / df)
            else:
                idf[w] = 0
        #Return IDF values to the dictionary
        return idf

    def get_tf(self) -> dict:
        """Get term frequencies (TF) for each document."""
          # Return a dictionary
        tf = {
            doc: {w: self.index[w][doc] for w in self.index if doc in self.index[w]} # Collect TF for each document
            for doc in self.doc_ids # Iterate over all unique document_IDs
        }
        return tf

    def get_log_tf(self, tf: dict) -> dict:
        """Apply log normalization to term frequencies (1 + log(tf))."""
        log_tf = {}
        for doc, terms in tf.items():
            #Create an empty dictionary for each document's log-normalized term frequencies
            log_tf[doc] = {}
            for w, freq in terms.items():
                #If frequency is bigger than 0, apply log normalization (1 + log(freq))
                log_tf[doc][w] = 1 + math.log(freq) if freq > 0 else 0
        return log_tf

    def get_ntf(self) -> dict:
        """Compute normalized term frequencies for each document."""
        ntf = {}
        a = 0.4  #Smoothing factor
        for doc, terms in self.tf.items():
            #Find the maximum term frequency in the document (to use for normalization)
            max_tf = max(terms.values()) if terms else 1 
            #Apply the normalization formula
            ntf[doc] = {w: a + (1 - a) * (freq / max_tf) for w, freq in terms.items()}
        #Return the value to the dictionary
        return ntf

    def get_doc_vector(self) -> dict:
        """Compute the magnitude of each document vector based on the term weighting scheme."""
        doc_vector = {}
        for doc, terms in self.tf.items():
            doc_vector[doc] = 0  
            for term, freq in terms.items():
                """
                If the weighting scheme is binary, set the weight to 1
                If the weighting scheme is tf, set the weight to the tf
                If the weighting scheme is tfidf, set the weight to (tf * idf)
                """
                if self.term_weighting == "binary":
                    weight = 1
                elif self.term_weighting == "tf":
                    weight = freq
                elif self.term_weighting == "tfidf":
                    weight = freq * self.idf.get(term, 0)
                #Add the squared weight to the document's vector sum
                doc_vector[doc] += weight ** 2
            #Compute the magnitude of the document vector
            doc_vector[doc] = math.sqrt(doc_vector[doc])  
        return doc_vector
    
    def get_query_vector(self, query_terms: list) -> dict:
        """Compute the query vector with appropriate weights."""
        query_vector = {}
        for w in query_terms:
            #Count the frequency of each term in the query
            query_vector[w] = query_vector.get(w, 0) + 1  
        """
        If the weighting scheme is binary, set each term's frequency in the query to 1
        If the weighting scheme is tfidf, multiply each term's frequency by its IDF
        """
        if self.term_weighting == "binary":
            for w in query_vector:
                query_vector[w] = 1
        elif self.term_weighting == "tfidf":
            for w in query_vector:
                if w in self.idf:
                    #Apply multiplication on the term frequency by the idf
                    query_vector[w] *= self.idf[w]
        return query_vector

    def for_query(self, query_terms: list) -> list:
        """Perform ranked retrieval for a single query and return a list of relevant doc IDs."""
        query_vector = self.get_query_vector(query_terms)
        #Get magnitude of the query vector
        query_magnitude = math.sqrt(sum(weight ** 2 for weight in query_vector.values()))

        scores = {}
        for doc in self.doc_ids:
            cosine_score = 0
            doc_vector = self.tf.get(doc, {})
            for w, q_weight in query_vector.items():
                """
                If the weighting scheme is binary, the document weight is 1
                If the weighting scheme is tf, the document weight is the term frequency in the document
                If the weighting scheme is tfidf, the document weight is the term frequency in the document multiplied by the IDF
                """
                if w in doc_vector:
                    if self.term_weighting == "binary": 
                        d_weight = 1
                    elif self.term_weighting == "tf":
                        d_weight = doc_vector[w]
                    elif self.term_weighting == "tfidf":
                        d_weight = doc_vector[w] * self.idf.get(w, 0)
                    #Update the cosine score
                    cosine_score += q_weight * d_weight  
            doc_magnitude = self.doc_vector.get(doc, 0)
            
            #If vectors for the query and document have non-zero magnitudes
            if query_magnitude > 0 and doc_magnitude > 0:  
                #Compute cosine similarity score
                scores[doc] = cosine_score / (query_magnitude * doc_magnitude)  # Divide the dot product by the magnitudes of vectors

        # Sort top 10 documents by their cosine scores
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        #Extract sorted document IDs
        sorted_docs = [doc_id for doc_id, _ in ranked_docs]
        return sorted_docs