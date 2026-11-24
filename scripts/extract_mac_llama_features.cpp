// Single-sequence action-span extraction for llama.cpp revision 620d2225c.
#include "llama-ext.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
constexpr int context_limit = 1024;

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct AlignedPrompt {
    std::vector<llama_token> ids;
    std::vector<int> positions;
};

static AlignedPrompt align(const llama_vocab * vocab, const std::string & prompt, size_t begin, size_t end) {
    require(begin < end && end < prompt.size(), "invalid action span");
    AlignedPrompt result;
    result.ids.resize(context_limit);
    const int count = llama_tokenize(vocab, prompt.data(), prompt.size(), result.ids.data(), context_limit, false, true);
    require(count > 0 && count <= context_limit, "prompt exceeds context budget");
    result.ids.resize(count);
    std::string reconstructed;
    for (int i = 0; i < count; ++i) {
        char piece[512];
        const int length = llama_token_to_piece(vocab, result.ids[i], piece, sizeof(piece), 0, true);
        require(length >= 0, "token piece exceeds buffer");
        const size_t start = reconstructed.size();
        reconstructed.append(piece, length);
        if (length && start < end && reconstructed.size() > begin) result.positions.push_back(i);
    }
    require(reconstructed == prompt && !result.positions.empty() && result.positions.back() < count - 1,
            "prompt or candidate could not be reconstructed from exact token bytes");
    return result;
}

static void write(std::ofstream & output, const std::vector<float> & vector) {
    require(std::all_of(vector.begin(), vector.end(), [](float x) { return std::isfinite(x); }), "nonfinite feature");
    output.write(reinterpret_cast<const char *>(vector.data()), vector.size() * sizeof(float));
    require(output.good(), "feature write failed");
}

int main(int argc, char ** argv) {
    try {
        require(argc == 4 || (argc == 5 && std::string(argv[4]) == "--metal"),
                "expected cached model, prompt directory, fresh output path, and optional --metal");
        ggml_backend_load_all();
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = argc == 5 ? -1 : 0;
        model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], model_params), llama_model_free);
        require(model != nullptr, "cached model load failed");
        auto params = llama_context_default_params();
        params.n_ctx = context_limit;
        params.n_batch = context_limit;
        params.n_ubatch = context_limit;
        params.n_threads = 4;
        params.n_threads_batch = 4;
        params.embeddings = true;
        params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), params), llama_free);
        require(context != nullptr, "embedding context initialization failed");
        const int layers = llama_model_n_layer(model.get()), width = llama_model_n_embd(model.get());
        require(width == llama_model_n_embd_out(model.get()), "output and block widths differ");
        for (int layer = 0; layer < layers; ++layer) llama_set_embeddings_layer_inp(context.get(), layer, true);
        const fs::path root = argv[2];
        std::ifstream list(root / "spans.tsv");
        std::ofstream output(argv[3], std::ios::binary);
        require(list.good() && output.good(), "could not open prompt list or output");
        int index;
        size_t begin, end;
        int seen = 0;
        while (list >> index >> begin >> end) {
            require(index == seen++, "spans must use increasing contiguous indices");
            std::ifstream file(root / (std::to_string(index) + ".prompt"), std::ios::binary);
            require(file.good(), "missing rendered prompt");
            const std::string prompt(std::istreambuf_iterator<char>(file), {});
            auto aligned = align(llama_model_get_vocab(model.get()), prompt, begin, end);
            llama_memory_clear(llama_get_memory(context.get()), true);
            const auto started = std::chrono::steady_clock::now();
            require(llama_decode(context.get(), llama_batch_get_one(aligned.ids.data(), aligned.ids.size())) == 0,
                    "decode failed");
            llama_synchronize(context.get());
            for (int pooling = 0; pooling < 2; ++pooling) {
                for (int layer = 0; layer <= layers; ++layer) {
                    const float * row_base = layer == layers ? nullptr : llama_get_embeddings_layer_inp(context.get(), layer);
                    require(layer == layers || row_base != nullptr, "missing block input");
                    const auto at = [&](int position) {
                        const float * row = layer == layers ? llama_get_embeddings_ith(context.get(), position)
                                                           : row_base + size_t(position) * width;
                        require(row != nullptr, "missing per-token final embedding");
                        return row;
                    };
                    std::vector<float> vector(width, 0.0f);
                    if (pooling == 0) {
                        std::copy_n(at(aligned.positions.back()), width, vector.begin());
                    } else {
                        for (const int pos : aligned.positions) {
                            const float * row = at(pos);
                            for (int j = 0; j < width; ++j) vector[j] += row[j];
                        }
                        for (auto & value : vector) value /= aligned.positions.size();
                    }
                    write(output, vector);
                }
            }
            std::cout << "{\"index\":" << index << ",\"tokens\":" << aligned.ids.size()
                      << ",\"layers\":" << layers + 1 << ",\"width\":" << width
                      << ",\"seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()
                      << ",\"action_positions\":[";
            for (size_t i = 0; i < aligned.positions.size(); ++i) std::cout << (i ? "," : "") << aligned.positions[i];
            std::cout << "],\"action_tokens\":[";
            for (size_t i = 0; i < aligned.positions.size(); ++i) std::cout << (i ? "," : "") << aligned.ids[aligned.positions[i]];
            std::cout << "]}" << std::endl;
        }
        require(list.eof() && seen > 0, "empty or malformed span index");
        context.reset();
        model.reset();
        llama_backend_free();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
