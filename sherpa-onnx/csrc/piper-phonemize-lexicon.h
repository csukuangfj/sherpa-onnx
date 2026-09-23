// sherpa-onnx/csrc/piper-phonemize-lexicon.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_PIPER_PHONEMIZE_LEXICON_H_
#define SHERPA_ONNX_CSRC_PIPER_PHONEMIZE_LEXICON_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "sherpa-onnx/csrc/offline-tts-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-kitten-model-meta-data.h"
#include "sherpa-onnx/csrc/offline-tts-kokoro-model-meta-data.h"
#include "sherpa-onnx/csrc/offline-tts-matcha-model-meta-data.h"
#include "sherpa-onnx/csrc/offline-tts-vits-model-meta-data.h"

namespace sherpa_onnx {

std::unordered_map<char32_t, int32_t> ReadPiperTokens(std::istream &is);

std::vector<int64_t> PiperPhonemesToIdsVits(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<char32_t> &phonemes, bool is_inflect);

std::vector<int64_t> CoquiPhonemesToIds(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<char32_t> &phonemes,
    const OfflineTtsVitsModelMetaData &vits_meta_data);

// Convert phoneme codepoints to token IDs for Matcha TTS models.
// Returns multiple sub-sequences if the input exceeds max_token_len.
std::vector<std::vector<int64_t>> PiperPhonemesToIdsMatcha(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<char32_t> &phonemes, bool use_eos_bos,
    int32_t max_token_len = 400);

// Convert phoneme codepoints to token IDs for Kitten TTS models.
// Returns multiple sub-sequences split by max_token_len.
std::vector<std::vector<int64_t>> PiperPhonemesToIdsKitten(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<char32_t> &phonemes,
    const OfflineTtsKittenModelMetaData &meta_data);

class PiperPhonemizeLexicon : public OfflineTtsFrontend {
 public:
  PiperPhonemizeLexicon(const std::string &tokens, const std::string &data_dir,
                        const OfflineTtsVitsModelMetaData &vits_meta_data);

  PiperPhonemizeLexicon(const std::string &tokens, const std::string &data_dir,
                        const OfflineTtsMatchaModelMetaData &matcha_meta_data);

  PiperPhonemizeLexicon(const std::string &tokens, const std::string &data_dir,
                        const OfflineTtsKokoroModelMetaData &kokoro_meta_data);

  PiperPhonemizeLexicon(const std::string &tokens, const std::string &data_dir,
                        const OfflineTtsKittenModelMetaData &kitten_meta_data);

  template <typename Manager>
  PiperPhonemizeLexicon(Manager *mgr, const std::string &tokens,
                        const std::string &data_dir,
                        const OfflineTtsVitsModelMetaData &vits_meta_data);

  template <typename Manager>
  PiperPhonemizeLexicon(Manager *mgr, const std::string &tokens,
                        const std::string &data_dir,
                        const OfflineTtsMatchaModelMetaData &matcha_meta_data);

  template <typename Manager>
  PiperPhonemizeLexicon(Manager *mgr, const std::string &tokens,
                        const std::string &data_dir,
                        const OfflineTtsKokoroModelMetaData &kokoro_meta_data);

  template <typename Manager>
  PiperPhonemizeLexicon(Manager *mgr, const std::string &tokens,
                        const std::string &data_dir,
                        const OfflineTtsKittenModelMetaData &kitten_meta_data);

  std::vector<TokenIDs> ConvertTextToTokenIds(
      const std::string &text, const std::string &voice = "") const override;

 private:
  std::vector<TokenIDs> ConvertTextToTokenIdsVits(
      const std::string &text, const std::string &voice = "") const;

  std::vector<TokenIDs> ConvertTextToTokenIdsMatcha(
      const std::string &text, const std::string &voice = "") const;

 private:
  // map unicode codepoint to an integer ID
  std::unordered_map<char32_t, int32_t> token2id_;
  OfflineTtsVitsModelMetaData vits_meta_data_;
  OfflineTtsMatchaModelMetaData matcha_meta_data_;
  OfflineTtsKokoroModelMetaData kokoro_meta_data_;
  OfflineTtsKittenModelMetaData kitten_meta_data_;
  bool is_matcha_ = false;
  bool is_kokoro_ = false;
  bool is_kitten_ = false;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_PIPER_PHONEMIZE_LEXICON_H_
