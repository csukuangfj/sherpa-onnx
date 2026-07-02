// Copyright (c)  2024  Xiaomi Corporation

#include <string>
#include <vector>

#include "macros.h"  // NOLINT
#include "napi.h"  // NOLINT

static Napi::Array ListRawFileDir(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  // This function is kept for API compatibility but always returns an empty
  // array since we no longer use the HarmonyOS raw file resource manager.
  return Napi::Array::New(env, 0);
}

void InitUtils(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "listRawfileDir"),
              Napi::Function::New(env, ListRawFileDir));
}
