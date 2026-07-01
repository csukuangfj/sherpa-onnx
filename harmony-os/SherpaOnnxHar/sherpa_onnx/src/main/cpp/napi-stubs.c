// Weak stubs for NAPI symbols that are declared (deprecated) in the OHOS SDK
// headers but not exported from libace_napi.z.so on API 10 and below.
// On API 11+ the real symbols from the shared library override these at runtime.
//
// NOTE: On API 10, napi_add_finalizer is a no-op, which means destructor
// callbacks registered by node-addon-api (e.g. via Napi::Function::New or
// Napi::External::New) will never fire. This causes minor memory leaks:
//   - Napi::Function::New callback data (~dozens of bytes per exported function,
//     allocated once at module load, lives for process lifetime anyway)
//   - Napi::External destructor won't be called by GC, so callers must
//     explicitly free native objects (our code already does this via
//     SherpaOnnxDestroyXxx calls from JavaScript)
// On API 11+ there is NO memory leak since the real implementation is used.

#include <stddef.h>
#include <stdint.h>

typedef struct napi_env__*napi_env;
typedef struct napi_value__*napi_value;
typedef int32_t napi_status;
typedef struct napi_async_context__*napi_async_context;
typedef struct napi_callback_scope__*napi_callback_scope;
typedef void (*napi_finalize)(napi_env env, void *finalize_data,
                              void *finalize_hint);
typedef struct napi_ref__*napi_ref;

__attribute__((weak)) napi_status
napi_add_finalizer(napi_env env, napi_value js_object, void *native_object,
                   napi_finalize finalize_cb, void *finalize_hint,
                   napi_ref *result) {
  (void)env;
  (void)js_object;
  (void)native_object;
  (void)finalize_cb;
  (void)finalize_hint;
  (void)result;
  return 0;
}

__attribute__((weak)) napi_status
napi_async_init(napi_env env, napi_value async_resource,
                napi_value async_resource_name,
                napi_async_context *result) {
  (void)env;
  (void)async_resource;
  (void)async_resource_name;
  if (result) *result = NULL;
  return 0;
}

__attribute__((weak)) napi_status
napi_async_destroy(napi_env env, napi_async_context async_context) {
  (void)env;
  (void)async_context;
  return 0;
}

__attribute__((weak)) napi_status
napi_open_callback_scope(napi_env env, napi_value resource_object,
                         napi_async_context context,
                         napi_callback_scope *result) {
  (void)env;
  (void)resource_object;
  (void)context;
  if (result) *result = NULL;
  return 0;
}

__attribute__((weak)) napi_status
napi_close_callback_scope(napi_env env, napi_callback_scope scope) {
  (void)env;
  (void)scope;
  return 0;
}
