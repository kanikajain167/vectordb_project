Vector Database from Scratch

This project is a vector database built completely from scratch using only NumPy, without using any existing library like FAISS, Pinecone, Chroma, or sklearn.neighbors. The goal was to actually understand how vector search works internally instead of just importing a library.

The project has two indexes. The first is a brute force exact index, which checks a query against every stored vector using cosine similarity, so it is always correct but slower. The second is an approximate index called IVF-Flat, which is built along with k-means clustering algorithm. It groups vectors into clusters and only searches the closest few clusters instead of the whole dataset, which makes it much faster. Both indexes are wrapped in one API with insert, search, and delete, so they stay in sync with each other.

To test this properly, I generated over 55,000 text sentences across 40 topics, converted them into vectors using TF-IDF and SVD, and set aside 500 queries with their correct answers precomputed. On 54,500 vectors, the approximate index is about 2x faster than brute force while still finding around 98.5% of the correct nearest neighbours on average.

I also wrote automated tests using pytest to check that insert, search, and delete are actually working correctly, not just tested by running the scripts and looking at the output.
