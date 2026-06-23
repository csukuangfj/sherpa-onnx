// sherpa-onnx/csrc/file-utils.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/file-utils.h"

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <limits.h>
#include <stdlib.h>
#endif

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

bool FileExists(const std::string &filename) {
  return std::ifstream(filename).good();
}

void AssertFileExists(const std::string &filename) {
  if (!FileExists(filename)) {
    SHERPA_ONNX_LOGE("filename '%s' does not exist", filename.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
}

std::vector<char> ReadFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return {};
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    return {};
  }

  return buffer;
}

#if __ANDROID_API__ >= 9
std::vector<char> ReadFile(AAssetManager *mgr, const std::string &filename) {
  if (!filename.empty() && filename[0] == '/') {
    SHERPA_ONNX_LOGE(
        "You are using an absolute path '%s', but assetManager is NOT set to "
        "null.",
        filename.c_str());

    SHERPA_ONNX_LOGE(
        "Please set assetManager to null when you load model files from the SD "
        "card");

    SHERPA_ONNX_LOGE(
        "See also https://github.com/k2-fsa/sherpa-onnx/issues/2562");
  }

  AAsset *asset = AAssetManager_open(mgr, filename.c_str(), AASSET_MODE_BUFFER);
  if (!asset) {
    __android_log_print(ANDROID_LOG_FATAL, "sherpa-onnx",
                        "Read binary file: Load '%s' failed", filename.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  auto p = reinterpret_cast<const char *>(AAsset_getBuffer(asset));
  size_t asset_length = AAsset_getLength(asset);

  std::vector<char> buffer(p, p + asset_length);
  AAsset_close(asset);

  return buffer;
}
#endif

#if __OHOS__
std::vector<char> ReadFile(NativeResourceManager *mgr,
                           const std::string &filename) {
  std::unique_ptr<RawFile, decltype(&OH_ResourceManager_CloseRawFile)> fp(
      OH_ResourceManager_OpenRawFile(mgr, filename.c_str()),
      OH_ResourceManager_CloseRawFile);

  if (!fp) {
    std::ostringstream os;
    os << "Read file '" << filename << "' failed.";
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
    return {};
  }

  auto len = static_cast<int32_t>(OH_ResourceManager_GetRawFileSize(fp.get()));

  std::vector<char> buffer(len);

  int32_t n = OH_ResourceManager_ReadRawFile(fp.get(), buffer.data(), len);

  if (n != len) {
    std::ostringstream os;
    os << "Read file '" << filename << "' failed. Number of bytes read: " << n
       << ". Expected bytes to read: " << len;
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
    return {};
  }

  return buffer;
}
#endif

std::string ResolveAbsolutePath(const std::string &path) {
  if (path.empty()) {
    return path;
  }

#ifdef _WIN32
  // Check if path is already absolute (drive letter or UNC path)
  if ((path.size() > 1 && path[1] == ':') ||
      (path.size() > 1 && path[0] == '\\' && path[1] == '\\')) {
    return path;
  }

  int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(),
                                 static_cast<int>(path.size()), nullptr, 0);
  if (wlen <= 0) {
    return path;  // fallback on failure
  }

  std::wstring wpath(wlen, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, path.c_str(), static_cast<int>(path.size()),
                      wpath.data(), wlen);

  wchar_t wbuffer[MAX_PATH];
  DWORD n = GetFullPathNameW(wpath.c_str(), MAX_PATH, wbuffer, nullptr);
  if (n == 0 || n >= MAX_PATH) {
    return path;  // fallback on failure
  }

  int ulen = WideCharToMultiByte(CP_UTF8, 0, wbuffer, static_cast<int>(n),
                                 nullptr, 0, nullptr, nullptr);
  if (ulen <= 0) {
    return path;  // fallback on failure
  }

  std::string result(ulen, '\0');
  WideCharToMultiByte(CP_UTF8, 0, wbuffer, static_cast<int>(n), result.data(),
                      ulen, nullptr, nullptr);
  return result;

#else
  // POSIX: absolute paths start with '/'
  if (path[0] == '/') {
    return path;
  }

  char buffer[PATH_MAX];
  if (realpath(path.c_str(), buffer)) {
    return std::string(buffer);
  }

  return path;  // fallback on failure
#endif
}

}  // namespace sherpa_onnx
