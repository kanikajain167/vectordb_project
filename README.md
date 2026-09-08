# Write the Vector Database Yourself - Project Submission

## About this project

This project is my submission for the task "Write the Vector Database
Yourself". The task was to not use any ready made vector database library
like Pinecone, FAISS, Chroma or sklearn.neighbors, and instead build one
myself using numpy so that I actually understand what happens inside a
vector database.

In this project I have built:

1. An exact brute force index - this checks the query against every single
   stored vector and gives the correct answer always. This is slow when
   data is large, but it is 100% correct, so I use it as the "ground
   truth" to check my other index against.
2. An approximate index called IVF-Flat, which I built from scratch. In
   this, first I group all the vectors into clusters using k-means
   (I wrote k-means myself too, it is not from sklearn), and when a query
   comes, I only search inside a few nearest clusters instead of the whole
   data. This makes search faster but it can miss some correct answers
   sometimes, that is why it is called "approximate".
3. A small API on top of both indexes with insert, search and delete
   functions, so both indexes stay in sync and can be compared directly.

I have generated my own text data (55,000+ short sentences from 40 different
topics like sports, cooking, finance, space etc) and converted it into
number vectors using TF-IDF + SVD (this is a real and commonly used text
embedding method, I did not use any random numbers for the vectors, the
text is real sentences so the vectors have real structure/clusters in
them, which is what the assignment asked for).

I have also written proper automated tests (using pytest) to check that
insert, search and delete are working correctly on both indexes, not just
tested by running scripts and looking at the screen.

I am a fresher and this is being submitted as my project work for
selection, so I have tried to explain everything in as simple way as
possible, in case someone less technical also has to check this.

---

## Folder structure - what each file/folder is for

```
vectordb_project/
|
|-- README.md                 <- this file
|-- requirements.txt          <- python libraries needed
|
|-- vectordb/                 <- the actual vector database code (the "product")
|   |-- __init__.py
|   |-- brute_force.py        <- exact index (ground truth)
|   |-- kmeans.py             <- k-means clustering, written from scratch
|   |-- ivf_flat.py           <- approximate index (IVF-Flat), built using kmeans.py
|   |-- api.py                <- VectorDB class, the single API over both indexes
|
|-- scripts/                  <- run these one by one, in this order, see below
|   |-- generate_data.py      <- STEP 1: makes the text data + vectors + test queries
|   |-- build_and_save.py     <- STEP 2: builds the actual indexes and saves them
|   |-- benchmark.py          <- STEP 3: compares exact vs approximate index
|   |-- demo_query.py         <- STEP 4: use this to actually try out the database
|
|-- tests/
|   |-- test_indexes.py       <- automated tests (pytest) for both indexes and the API
|
|-- data/            <- NOT included in the zip, gets created when you run generate_data.py
|-- artifacts/        <- NOT included in the zip, gets created when you run build_and_save.py
|-- results/          <- NOT included in the zip, gets created when you run benchmark.py
```

I have not included the `data/`, `artifacts/` and `results/` folders in
the zip file on purpose, because they get auto-created when you run the
scripts, and including them would just make the zip file unnecessarily
big (the embeddings file alone is around 27 MB). So please run the steps
below, they will create these folders automatically, it does not need any
manual folder creation from your side.

---

## How to run this project (do these steps in exact order)

### Step 0: Setup (only needed one time)

Open a terminal, go inside the project folder, and run:

```
pip install -r requirements.txt
```

This will install numpy, scikit-learn and pytest. I have used
scikit-learn ONLY for TF-IDF and SVD, which are used to turn text
sentences into number vectors (this is a text preprocessing step, this is
NOT the vector database part). The actual vector database - the brute
force search, the k-means clustering, the IVF-Flat index, insert/search/
delete - all of that is written by me using plain numpy only, no vector
database library and no sklearn.neighbors is used anywhere, as the
assignment asked.

### Step 1: Generate the data

```
python scripts/generate_data.py
```

What this does:
- Creates 55,000+ short text sentences from 40 topics (sports, cooking,
  finance, medicine, space etc). I generated these myself using
  templates and topic word-lists, because I did not want to depend on
  downloading any dataset from the internet (also some environments
  don't allow internet access for that, so this way it always works).
- Converts every sentence into a 128-number vector using TF-IDF + SVD.
- Picks out 500 sentences to the side to use as test queries later, and
  computes their EXACT top-10 nearest neighbours using brute force. This
  becomes the "correct answer key" that I check my approximate index
  against, later in Step 3.
- Saves everything inside a new `data/` folder.

This takes about 10-15 seconds to run on a normal laptop.

### Step 2: Build the indexes

```
python scripts/build_and_save.py
```

What this does:
- Loads the vectors created in Step 1.
- Trains the IVF-Flat index (this means running k-means to make 256
  clusters out of the 54,500 vectors).
- Loads all vectors into both the exact index and the approximate index.
- Saves the whole built database into `artifacts/vector_db.pkl` so we
  don't have to rebuild it again and again every time.

This takes about 5-10 seconds to run.

### Step 3: Run the benchmark (this is the actual "point" of the assignment)

```
python scripts/benchmark.py
```

What this does:
- Loads the saved database from Step 2.
- Runs all 500 test queries through the exact index and notes how long it
  takes.
- Runs the same 500 queries through the approximate index and notes how
  long that takes.
- Compares the approximate index's answers against the correct answer key
  from Step 1, and calculates something called "Recall@10" - this basically
  means, out of the 10 correct nearest neighbours, how many did the
  approximate index actually manage to find.
- Prints a full report on screen, and also saves it to
  `results/benchmark_report.txt`.

On my machine I got approximately these numbers (may be little different
on your machine, that is normal):

```
Vectors indexed        : 54500
Exact  (brute force)    : mean ~1.3 ms per query
Approx (IVF-Flat)       : mean ~0.65 ms per query
Speedup                 : around 2x faster
Recall@10 mean          : around 0.985 (98.5%)
```

So the approximate index is roughly 2 times faster than brute force, but
it still finds about 98.5% of the true correct neighbours on average. This
is basically the whole tradeoff the assignment wanted to show - you give
up a small bit of accuracy to get speed, and now I can actually show the
exact numbers for that tradeoff instead of just saying it.

(Note: with only 54,500 vectors of 128 dimensions, brute force itself is
already quite fast on a modern CPU, that's why the speedup is "only" 2x
here and not 10x or 50x. On much bigger datasets - millions of vectors -
the gap between brute force and IVF-Flat would be a lot bigger, because
brute force time grows directly with the number of vectors, while
IVF-Flat only searches a few clusters no matter how big the data gets.)

### Step 4: Try it yourself / demo

To search for anything you like, just type any sentence:

```
python scripts/demo_query.py
```

This opens an interactive prompt, type any sentence and press enter, it
will show you the top 10 closest matching sentences from the corpus, from
both the exact index and the approximate index side by side, along with
how long each one took. Type `exit` to quit.

You can also do a single one-line query without going interactive:

```
python scripts/demo_query.py --query "the striker scored a goal in the final match"
```

To see insert and delete actually working (this is for showing the CRUD
part of the API, "insert, search top-k, delete" as mentioned in the
assignment), run:

```
python scripts/demo_query.py --live-demo
```

This single command will, in this order:
1. Insert a brand new sentence that was never in the original data.
2. Search for it and show it comes back as the #1 result with a perfect
   match score of 1.0.
3. Delete that same sentence from both indexes.
4. Search again and prove that it is now gone from the results.

This is the easiest single command to run and record for a demo video,
since it proves insert, search AND delete are all working, all in one go.

If you want to insert or delete something and have it saved permanently
(so it stays there even after you close the terminal), add `--save`:

```
python scripts/demo_query.py --insert "The robot vacuum cleaned the living room." --save
python scripts/demo_query.py --delete 1234 --save
```

(Without `--save` the change only happens for that one run, it will not
be written back to the saved database file. I did it this way on purpose
so that running the demo again and again doesn't keep changing the data
file every single time by mistake.)

### Step 5 (optional but recommended): Run the automated tests

```
pytest tests/ -v
```

This runs 15 automated tests I wrote which check things like:
- does searching for a stored vector actually return that same vector
- does the brute force ranking match a manually calculated correct ranking
- does delete actually remove the vector so it stops showing up in results
- does delete raise a proper error if you try to delete something that
  doesn't exist
- does inserting after a delete work correctly
- does the IVF-Flat index basically match brute force when it is allowed
  to search all clusters (nprobe = nlist)
- does the combined VectorDB API keep both indexes in sync for insert and
  delete

All 15 tests were passing when I submitted this.

---

## A few design decisions I want to explain (in case asked)

**Why cosine similarity and not plain Euclidean distance?**
I normalise every vector to length 1 before storing it (this is called
L2 normalisation). Once vectors are normalised, cosine similarity between
two vectors becomes the same as just their dot product, which is a single
fast matrix multiplication in numpy. This is why the search functions in
this project look so simple, most of the "cosine similarity math" is
already handled by the normalising step.

**Why did I use TF-IDF + SVD for the text embeddings and not something
like word2vec or a proper sentence transformer model?**
TF-IDF + SVD (this combination is called LSA) is a real, standard and
well known text embedding technique, and it doesn't need to download any
big pretrained model from the internet, which means this project can be
run by anyone anywhere without depending on internet access or huge model
downloads. It's not the most powerful embedding out there, but it gives
real vectors with real topic-based clusters in them, which is exactly
what the assignment needed to make the IVF-Flat clustering meaningful
(the assignment itself said real embeddings are "lumpy" and that's more
interesting to test against than pure random vectors, that's why I did
not just use plain random numpy vectors).

**Why did I pick IVF-Flat and not HNSW?**
The assignment said IVF-Flat is the "gentler path" and HNSW is "harder
and more impressive" but either is acceptable. Since this is for a
selection task and I wanted to make sure I deliver something fully
correct and fully tested rather than something half-working and
impressive-looking, I picked IVF-Flat and made sure it is implemented
properly, trained properly (with my own k-means, not sklearn's), and
benchmarked properly against the exact index.

**About delete in the approximate index, and why the assignment says
delete is "genuinely awkward" in these kind of indexes:**
In IVF-Flat, every vector belongs to one cluster. I store each cluster as
a Python dictionary of `{id: vector}` instead of one packed numpy matrix
per cluster. This was a deliberate choice: with a dictionary, deleting a
vector is a simple, cheap operation (just remove that one key). If I had
instead stored each cluster as one single numpy array (which is a little
faster to search), then deleting even one vector would mean rebuilding
that entire cluster's array from scratch every single time, which gets
slow and messy as the cluster grows. So I accepted a small extra cost at
search time (I have to stack the dictionary's vectors into a temporary
array just for that search) in exchange for keeping delete simple and
fast. I have written this exact explanation as a comment inside
`vectordb/ivf_flat.py` as well, above the `delete()` function.

**Why not use FAISS/Pinecone/Chroma/sklearn.neighbors?**
Because the assignment specifically said not to. Every part of the
similarity search, clustering, insert, search and delete logic in the
`vectordb/` folder is written by me using plain numpy. The only external
library used anywhere is scikit-learn's `TfidfVectorizer` and
`TruncatedSVD`, and that is only used in `scripts/generate_data.py` to
turn text sentences into vectors in the first place - it has nothing to
do with the actual vector database logic, which starts only after the
vectors already exist.

---

## Things I would improve if I had more time

- Right now IVF-Flat uses fixed values for the number of clusters (256)
  and number of clusters searched per query (8). With more time I would
  add a script that tries a few different values of these and plots
  recall vs speed, so it's easier to pick the best setting for a given
  use case.
- I would also like to try implementing HNSW as a second approximate
  index option, since the assignment mentioned it as the more advanced
  path, but I ran out of time to do it properly and test it as
  thoroughly as I wanted to for this submission.
- Right now if you close the terminal without using `--save`, insert and
  delete changes are lost. A more production ready version would
  auto-save changes, or use a proper file-based storage instead of one
  pickle file for the whole database.

---

Thank you for checking my submission.
