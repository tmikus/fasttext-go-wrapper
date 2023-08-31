/**
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <unistd.h>
#include <iostream>
#include <istream>
#include <sstream>
#include <vector>
#include <cstring>
#include <fastText/fasttext.h>
#include <fastText/autotune.h>
#include <fasttext-wrapper.hpp>

extern "C" {

    fasttext::FastText ft_model;
    bool ft_initialized = false;

    bool ft_has_newline(std::string str) {
        return (0 == str.compare(str.length() - 1, 1, "\n"));
    };

    int ft_load_model(const char *path) {
        if (!ft_initialized) {
            if(access(path, F_OK) != 0) {
                return -1;
            }
            ft_model.loadModel(std::string(path));
            ft_initialized = true;
        }
        return 0;
    }

    go_fast_text_pair_t* ft_predict(const char *query_in, int k, float threshold, int* result_length)
    {
        std::string query(query_in);

        if (!ft_has_newline(query)) {
            query.append("\n");
        }

        std::istringstream inquery(query);
        std::istream &in = inquery;

        std::vector<std::pair<fasttext::real, std::string>> predictions;

        ft_model.predictLine(in, predictions, k, threshold);

        int result_size = predictions.size();

        go_fast_text_pair_t* pairsArray = (go_fast_text_pair_t*) malloc(result_size * sizeof(go_fast_text_pair_t));

        for (int i = 0; i < int(predictions.size()); i++){
            const std::string::size_type label_size = predictions[i].second.size();
            pairsArray[i].label = new char[label_size + 1];
            memcpy(pairsArray[i].label, predictions[i].second.c_str(), label_size + 1);
            pairsArray[i].prob = predictions[i].first;
        }
        *result_length = result_size;

        return pairsArray;
    }

    int ft_get_vector_dimension()
    {
        if(!ft_initialized) {
            return -1;
        }
        return ft_model.getDimension();
    }

    int ft_get_sentence_vector(const char* query_in, float* vector_out, int vector_size)
    {
        std::string query(query_in);

        if(!ft_has_newline(query)) {
            query.append("\n");
        }

        std::istringstream inquery(query);
        std::istream &in = inquery;
        fasttext::Vector svec(ft_model.getDimension());

        ft_model.getSentenceVector(in, svec);
        if(svec.size() != vector_size) {
            return -1;
        }
        memcpy(vector_out, svec.data(), vector_size*sizeof(float));
        return 0;
    }

    int ft_save_model(const char* filename)
    {
        if (!ft_initialized) {
            if (access(filename, W_OK) != 0) {
                return -1;
            }
        }
        ft_model.saveModel(std::string(filename));
        return 0;
    }

    // int ft_train(const char* model_name, const char* input, const char* output, int epoch, int word_ngrams, int thread, float lr)
    // {
    //     std::shared_ptr<fasttext::FastText> fasttext = std::make_shared<fasttext::FastText>();
    //     fasttext::Args a = fasttext::Args();
        
    //     if (std::string(model_name) == "supervised") {
    //         a.model = fasttext::model_name::sup;
    //     } else if (std::string(model_name) == "cbow") {
    //         a.model = fasttext::model_name::cbow;
    //     } else if (std::string(model_name) == "skipgram") {
    //         a.model = fasttext::model_name::sg;
    //     } else {
    //         return -1;
    //     }
            
    //     a.input = std::string(input);
    //     a.output = std::string(output);
    //     a.epoch = epoch;
    //     a.wordNgrams = word_ngrams;
    //     a.thread = thread;
    //     a.lr = lr;

    //     std::ofstream ofs(output);
    //     if (!ofs.is_open()) {
    //         throw std::invalid_argument(std::string(output) + " cannot be opened for saving.");
    //     }
    //     ofs.close();

    //     fasttext->train(a);
    //     fasttext->saveModel(output);
    //     return 0;
    // }

    int train(const char* model_name, const char* input, const char* output, int epoch, int word_ngrams, int thread, float lr) {
        const std::vector<std::string> args = {
            "fasttext",
            std::string(model_name),
            "-input",
            std::string(input),
            "-output",
            std::string(output),
            "-epoch",
            std::to_string(epoch),
            "-wordNgrams",
            std::to_string(word_ngrams),
            "-thread",
            std::to_string(thread),
            "-lr",
            std::to_string(lr)
        };

        fasttext::Args a = fasttext::Args();
        a.parseArgs(args);
        std::shared_ptr<fasttext::FastText> fasttext = std::make_shared<fasttext::FastText>();
        std::string outputFileName;

        if (a.hasAutotune() &&
            a.getAutotuneModelSize() != fasttext::Args::kUnlimitedModelSize) {
            outputFileName = a.output + ".ftz";
        } else {
            outputFileName = a.output + ".bin";
        }
        std::ofstream ofs(outputFileName);
        if (!ofs.is_open()) {
            throw std::invalid_argument(
                outputFileName + " cannot be opened for saving.");
        }
        ofs.close();
        if (a.hasAutotune()) {
            fasttext::Autotune autotune(fasttext);
            autotune.train(a);
        } else {
            fasttext->train(a);
        }
        fasttext->saveModel(outputFileName);
        fasttext->saveVectors(a.output + ".vec");
        if (a.saveOutput) {
            fasttext->saveOutput(a.output + ".output");
        }
        return 0;
    }


    int quantize(const char* input, const char* output) {
        const std::vector<std::string> args = {
            "fasttext",
            "quantize",
            "-input",
            std::string(input),
            "-output",
            std::string(output),
        };
        fasttext::Args a = fasttext::Args();
        if (args.size() < 3) {
           return 1;
        }
        a.parseArgs(args);
        fasttext::FastText fasttext;
        fasttext.loadModel(a.output + ".bin");
        fasttext.quantize(a);
        fasttext.saveModel(a.output + ".ftz");
        return 0;
    }
}
