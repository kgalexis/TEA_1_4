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
