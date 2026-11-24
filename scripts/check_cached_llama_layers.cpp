// Technical smoke for the pinned llama.cpp 0.7.0 ABI. No study data is loaded.
#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static void write_vector(std::ofstream & out, const std::vector<float> & vector) {
    require(std::all_of(vector.begin(), vector.end(), [](float x) { return std::isfinite(x); }), "nonfinite vector");
    out.write(reinterpret_cast<const char *>(vector.data()), vector.size() * sizeof(float));
    require(out.good(), "feature write failed");
}

static void check_pair(llama_model * model, bool recycle, std::ofstream & output) {
    std::string prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "Synthetic only. Recycle: ";
    prompt += recycle ? "enabled" : "disabled";
    prompt += ". Candidate action: ";
    const std::string action = R"({"tool":"delete_file","arguments":{"path":"fixture.txt"}})";
    const size_t begin = prompt.size(), end = begin + action.size();
    prompt += action + "\nAssess the candidate.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    const auto * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(256);
    const int count = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), false, true);
    require(count > 0 && count < 256, "token or context budget exceeded");
    tokens.resize(count);

    std::vector<int> positions;
    std::string reconstructed;
    for (int i = 0; i < count; ++i) {
        char piece[512];
        const int n = llama_token_to_piece(vocab, tokens[i], piece, sizeof(piece), 0, true);
        require(n >= 0, "token piece too large");
        const size_t start = reconstructed.size();
        reconstructed.append(piece, n);
        if (start < end && reconstructed.size() > begin) positions.push_back(i);
    }
    require(reconstructed == prompt && !positions.empty() && positions.back() < count - 1, "action span mismatch");

    auto params = llama_context_default_params();
    params.n_ctx = 256;
    params.n_batch = 256;
    params.n_ubatch = 256;
    params.n_threads = 4;
    params.n_threads_batch = 4;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model, params), llama_free);
    require(context != nullptr, "could not create context");
    const auto decode = [&] {
        require(llama_decode(context.get(), llama_batch_get_one(tokens.data(), count)) == 0, "decode failed");
        llama_synchronize(context.get());
        auto * logits = llama_get_logits_ith(context.get(), -1);
        require(logits != nullptr, "missing logits");
        return logits;
    };
    const auto * initial = decode();
    const std::vector<float> reference(initial, initial + llama_vocab_n_tokens(vocab));
    const int layers = llama_model_n_layer(model), width = llama_model_n_embd(model);
    llama_memory_clear(llama_get_memory(context.get()), true);
    for (int layer = 0; layer < layers; ++layer) llama_set_embeddings_layer_inp(context.get(), layer, true);
    const auto * instrumented = decode();
    float largest_difference = 0.0f;
    for (size_t i = 0; i < reference.size(); ++i) {
        require(std::isfinite(reference[i]) && std::isfinite(instrumented[i]), "nonfinite logits");
        largest_difference = std::max(largest_difference, std::abs(reference[i] - instrumented[i]));
    }
    require(largest_difference <= 1e-4f, "instrumentation changed logits");
    for (int pool = 0; pool < 2; ++pool) {
        for (int layer = 0; layer < layers; ++layer) {
            const auto * rows = llama_get_embeddings_layer_inp(context.get(), layer);
            require(rows != nullptr, "missing block input");
            std::vector<float> vector(width, 0.0f);
            if (pool == 0) {
                std::copy_n(rows + size_t(positions.back()) * width, width, vector.begin());
            } else {
                for (const int pos : positions) {
                    for (int i = 0; i < width; ++i) vector[i] += rows[size_t(pos) * width + i];
                }
                for (float & value : vector) value /= positions.size();
            }
            write_vector(output, vector);
        }
    }
    std::cout << "{\"recycle\":" << (recycle ? "true" : "false") << ",\"prompt_tokens\":" << count
              << ",\"action_positions\":[";
    for (size_t i = 0; i < positions.size(); ++i) std::cout << (i ? "," : "") << positions[i];
    std::cout << "],\"action_token_ids\":[";
    for (size_t i = 0; i < positions.size(); ++i) std::cout << (i ? "," : "") << tokens[positions[i]];
    std::cout << "],\"block_inputs\":" << layers << ",\"width\":" << width
              << ",\"max_abs_logits_difference\":" << largest_difference << "}\n";
}

int main(int argc, char ** argv) {
    try {
        require(argc == 3, "expected cached GGUF and fresh feature path");
        ggml_backend_load_all();
        llama_backend_init();
        auto params = llama_model_default_params();
        params.n_gpu_layers = 0;
        params.load_mode = LLAMA_LOAD_MODE_MMAP;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], params), llama_model_free);
        require(model != nullptr, "cached model load failed");
        std::ofstream output(argv[2], std::ios::binary);
        require(output.good(), "could not create feature file");
        check_pair(model.get(), true, output);
        check_pair(model.get(), false, output);
        model.reset();
        llama_backend_free();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
