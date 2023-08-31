/**
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char* label;
    float prob;                                                                                         
} go_fast_text_pair_t;

/**
 * Initialize the fasttext model located on `path`
 * returns 0 on success
 */
int ft_load_model(const char* path);

/**
 * Predict a given keyword
 * `query_in`: The actual keyword to predict
 * `prob`: floating value to determine the probability of the result
 * `out`: Predicted value
 * `out_size`: How much characters to be copied into `out`
 * returns 0 on success
 */
// int ft_predict(const char* query_in, float* prob, char* out, int out_size);

/**
 * **/
go_fast_text_pair_t* ft_predict(const char *query_in, int k, float threshold, int* result_length);

/**
 * get dimension of vector from loaded model
 * returns positive dimension on success
 */
int ft_get_vector_dimension();

/**
 * get vector representation from given sentence
 * `query_in`: The actual keyword to predict
 * `vector`: PRE-ALLOCATED buffer for sentence vector
 * `vector_size`: dimension of allocated vector
 * returns 0 on success
 */
int ft_get_sentence_vector(const char* query_in, float* vector, int vector_size);

/**
 * train function training fasttext model from input file
 * `model_name`: type of training model "skipgram", "cbow", "supervised"
 * `input`: input filename
 * `output`: output filename
 * `epoch`: epoch number
 * `word_ngrams`: n gram range parameter (1, word_ngrams)
 * `thread`: thread number
 * `lr`: learning rate of the algorithm
 * returns 0 on success
 */
int train(const char* model_name, const char* input, const char* output, int epoch, int word_ngrams, int thread, float lr);

/**
 * quantize function zip fasttext model from
 * `input`: input filename to
 * `output`: output filename
 * returns 0 on success
 */
int quantize(const char* input, const char* output);

/**
 * save model to given file
 * `filename`: the name of the file
 * returns 0 on success
 **/ 
int ft_save_model(const char* filename);


#ifdef __cplusplus
}
#endif
