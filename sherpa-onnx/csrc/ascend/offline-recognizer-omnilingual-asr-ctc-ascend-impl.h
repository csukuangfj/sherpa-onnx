// sherpa-onnx/csrc/ascend/offline-recognizer-omnilingual-asr-ctc-ascend-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ASCEND_OFFLINE_RECOGNIZER_OMNILINGUAL_ASR_CTC_ASCEND_IMPL_H_
#define SHERPA_ONNX_CSRC_ASCEND_OFFLINE_RECOGNIZER_OMNILINGUAL_ASR_CTC_ASCEND_IMPL_H_

#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/ascend/offline-omnilingual-asr-ctc-model-ascend.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/rknn/offline-ctc-greedy-search-decoder-rknn.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

// defined in ../offline-recognizer-ctc-impl.h
OfflineRecognitionResult Convert(const OfflineCtcDecoderResult &src,
                                 const SymbolTable &sym_table,
                                 int32_t frame_shift_ms,
                                 int32_t subsampling_factor);

class OfflineRecognizerOmnilingualAsrCtcAscendImpl
    : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerOmnilingualAsrCtcAscendImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineOmnilingualAsrCtcModelAscend>(
            config.model_config)) {
    // For omnilingual ASR models, blank id is 0
    int32_t blank_id;
    if (config.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OfflineCtcGreedySearchDecoderRknn>(blank_id);
    } else {
      SHERPA_ONNX_LOGE("Only greedy_search is supported at present. Given %s",
                       config.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  template <typename Manager>
  OfflineRecognizerOmnilingualAsrCtcAscendImpl(
      Manager *mgr, const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineOmnilingualAsrCtcModelAscend>(
            mgr, config.model_config)) {
    // for omnilingual ASR CTC model, blank id is 0
    int32_t blank_id = 0;
    if (config.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OfflineCtcGreedySearchDecoderRknn>(blank_id);
    } else {
      SHERPA_ONNX_LOGE("Only greedy_search is supported at present. Given %s",
                       config.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(OmnilingualAsrTag{});
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i < n; ++i) {
      DecodeOneStream(ss[i]);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void DecodeOneStream(OfflineStream *s) const {
    std::vector<float> samples = s->GetFrames();

    std::vector<float> logits = model_->Run(std::move(samples));
    int32_t vocab_size = model_->VocabSize();
    int32_t num_out_frames = logits.size() / vocab_size;

    auto result = decoder_->Decode(logits.data(), num_out_frames, vocab_size);

    int32_t frame_shift_ms = 20;
    int32_t subsampling_factor = 1;
    auto r = Convert(result, symbol_table_, frame_shift_ms, subsampling_factor);

    r.text = ApplyInverseTextNormalization(std::move(r.text));
    r.text = ApplyHomophoneReplacer(std::move(r.text));
    s->SetResult(r);
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineOmnilingualAsrCtcModelAscend> model_;
  std::unique_ptr<OfflineCtcGreedySearchDecoderRknn> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ASCEND_OFFLINE_RECOGNIZER_OMNILINGUAL_ASR_CTC_ASCEND_IMPL_H_
