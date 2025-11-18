// sherpa-onnx/csrc/ascend/offline-omnilingual-asr-ctc-model-ascend.cc
//
// Copyright (c)  2025  Xiaomi Corporation

// References:
// https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/appdevgapi/aclcppdevg_03_0298.html
#include "sherpa-onnx/csrc/ascend/offline-omnilingual-asr-ctc-model-ascend.h"

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/ascend/macros.h"
#include "sherpa-onnx/csrc/ascend/utils.h"
#include "sherpa-onnx/csrc/file-utils.h"

namespace sherpa_onnx {

class OfflineOmnilingualAsrCtcModelAscend::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    PreInit();
    InitModel(config_.omnilingual.model);
    PostInit();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    PreInit();
    {
      auto buf = ReadFile(mgr, config_.omnilingual.model);
      InitModel(buf.data(), buf.size());
    }
    PostInit();
  }

  std::vector<float> Run(std::vector<float> samples) {
    // TODO(fangjun): Support multi clients
    std::lock_guard<std::mutex> lock(mutex_);

    int32_t num_samples = samples.size();
    if (num_samples >= max_num_samples_) {
      SHERPA_ONNX_LOGE("Set samples from %d to %d", num_samples,
                       max_num_samples_);
      num_samples = max_num_samples_;
    }

    aclError ret =
        aclrtMemcpy(*samples_ptr_, num_samples * sizeof(float), samples.data(),
                    num_samples * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    AclMdlDataset input_dataset;
    AclDataBuffer samples_buf(*samples_ptr_, num_samples * sizeof(float));
    input_dataset.AddBuffer(samples_buf);

    // dynamic shape input
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/appdevg/acldevg/aclcppdevg_000044.html

    std::array<int64_t, 2> samples_shape = {1, num_samples};
    AclTensorDesc samples_desc(ACL_FLOAT, samples_shape.size(),
                               samples_shape.data(), ACL_FORMAT_ND);
    input_dataset.SetTensorDesc(samples_desc, 0);

    AclMdlDataset output_dataset;

    int32_t num_frames = num_samples / 320;

    AclDataBuffer logits_buf(*logits_ptr_,
                             num_frames * vocab_size_ * sizeof(float));
    output_dataset.AddBuffer(logits_buf);

    ret = aclmdlExecute(*model_, input_dataset, output_dataset);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclmdlExecute");

    std::vector<float> logits(num_frames * vocab_size_);
    ret = aclrtMemcpy(logits.data(), num_frames * vocab_size_ * sizeof(float),
                      *logits_ptr_, num_frames * vocab_size_ * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    return logits;
  }

  int32_t VocabSize() const { return vocab_size_; }

 private:
  void InitModel(const std::string &filename) {
    model_ = std::make_unique<AclModel>(filename);
    if (config_.debug) {
      auto s = model_->GetInfo();
      SHERPA_ONNX_LOGE("%s", s.c_str());
    }
  }

  void InitModel(void *data, size_t size) {
    model_ = std::make_unique<AclModel>(data, size);
    if (config_.debug) {
      auto s = model_->GetInfo();
      SHERPA_ONNX_LOGE("%s", s.c_str());
    }
  }

  void PreInit() {
    int32_t device_id = 0;
    aclError ret = aclrtSetDevice(device_id);
    SHERPA_ONNX_ASCEND_CHECK(
        ret, "Failed to call aclrtSetDevice with device id: %d", device_id);

    context_ = std::make_unique<AclContext>(device_id);

    ret = aclrtSetCurrentContext(*context_);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtSetCurrentContext");
  }

  void PostInit() {
    vocab_size_ = model_->GetOutputShapes()[0].back();

    int32_t s = model_->GetInputShapes()[0].back();
    if (s != -1) {
      SHERPA_ONNX_LOGE("set max num samples to %d", s);
      max_num_samples_ = s;
    }

    Preallocate();
  }

  void Preallocate() {
    samples_ptr_ =
        std::make_unique<AclDevicePtr>(max_num_samples_ * sizeof(float));

    // stride is 20ms, sample rate is 16000
    // 20ms is 320 samples
    logits_ptr_ = std::make_unique<AclDevicePtr>(
        ((max_num_samples_ / 320) + 2) * vocab_size_ * sizeof(float));
    // +2 for over allocation
  }

 private:
  std::mutex mutex_;
  Acl acl_;

  std::unique_ptr<AclContext> context_;

  OfflineModelConfig config_;

  std::unique_ptr<AclModel> model_;
  int32_t vocab_size_ = 0;
  int32_t max_num_frames_ = 0;
  int32_t max_num_samples_ = 40 * 16000;  // 40 seconds
  int32_t feat_dim_ = 560;

  std::unique_ptr<AclDevicePtr> samples_ptr_;
  std::unique_ptr<AclDevicePtr> logits_ptr_;
};

OfflineOmnilingualAsrCtcModelAscend::OfflineOmnilingualAsrCtcModelAscend(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineOmnilingualAsrCtcModelAscend::OfflineOmnilingualAsrCtcModelAscend(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineOmnilingualAsrCtcModelAscend::~OfflineOmnilingualAsrCtcModelAscend() =
    default;

std::vector<float> OfflineOmnilingualAsrCtcModelAscend::Run(
    std::vector<float> samples) const {
  return impl_->Run(std::move(samples));
}

int32_t OfflineOmnilingualAsrCtcModelAscend::VocabSize() const {
  return impl_->VocabSize();
}

#if __ANDROID_API__ >= 9
template OfflineOmnilingualAsrCtcModelAscend::
    OfflineOmnilingualAsrCtcModelAscend(AAssetManager *mgr,
                                        const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineOmnilingualAsrCtcModelAscend::
    OfflineOmnilingualAsrCtcModelAscend(NativeResourceManager *mgr,
                                        const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
