Hi folks,

I just made some minor edits (rewritting and renaming) in order to get a cleaner code. I also added the *UNK* idea.

1. lim is not anymore class field.
2. add '_' for pseudo-private functions
3. bag --> vocabulary(class field) + print_bag() removed
4. changed _read_corpus_file() in order to close the corpus file
5. changed $START$, $END$ to *START*, *END* as it is in the assignment
6. changed _train() in order to keep only what is needed + *UNK* tokens implemented
7. changed test() interface, removed smoothing as this is just a Laplace model
8. prob() renamed as _get_probability() including the code of n_count, d_count for simplicity
9. score() --> entropy(), then perplexity() can be implemented by calling entropy() QUESTION: can entropy be computed on a list of test sentences or just on one sentence??? (see slide 40)



Dimitris:

Hi folks,

I just made some minor edits to include the perplexity measure and the selection of random words for the test sentences

1. Created a train_dataset and a test_dataset from the initial text
2. Added test_sentence_splitting function to get a list with all the sentences of the test dataset after preprocessing. We use this list later to get random (correct) sentences of the the test dataset.
3. Added get_test_sentence function to get a random (correct) sentence from the test dataset.
4. Added create_random_sentence which creates a random sentence of selected length(in words) from the test_set vocabulary words excluding the special words(words that contain "*" character).
5. Changed entropy function name to eval_measures and added perplexity computation using the first formula of page 41 in the slides which uses entropy for the computation of the perplexity.
6. Finally, we use 1 correct sentence of the test dataset and 4 sentences(equal length in words with the first) of random words of the test dataset for the evaluation of each n-gram. 
