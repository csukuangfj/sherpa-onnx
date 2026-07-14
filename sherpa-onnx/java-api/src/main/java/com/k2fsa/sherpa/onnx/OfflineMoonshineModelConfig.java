// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineMoonshineModelConfig {
    private final String preprocessor;
    private final String encoder;
    private final String uncachedDecoder;
    private final String cachedDecoder;
    private final String mergedDecoder;
    private final String decoder;

    private OfflineMoonshineModelConfig(Builder builder) {
        this.preprocessor = builder.preprocessor;
        this.encoder = builder.encoder;
        this.uncachedDecoder = builder.uncachedDecoder;
        this.cachedDecoder = builder.cachedDecoder;
        this.mergedDecoder = builder.mergedDecoder;
        this.decoder = builder.decoder;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getPreprocessor() {
        return preprocessor;
    }

    public String getEncoder() {
        return encoder;
    }

    public String getUncachedDecoder() {
        return uncachedDecoder;
    }

    public String getCachedDecoder() {
        return cachedDecoder;
    }

    public String getMergedDecoder() {
        return mergedDecoder;
    }

    public String getDecoder() {
        return decoder;
    }

    public static class Builder {
        private String preprocessor = "";
        private String encoder = "";
        private String uncachedDecoder = "";
        private String cachedDecoder = "";
        private String mergedDecoder = "";
        private String decoder = "";

        public OfflineMoonshineModelConfig build() {
            return new OfflineMoonshineModelConfig(this);
        }

        public Builder setPreprocessor(String preprocessor) {
            this.preprocessor = preprocessor;
            return this;
        }

        public Builder setEncoder(String encoder) {
            this.encoder = encoder;
            return this;
        }

        public Builder setUncachedDecoder(String uncachedDecoder) {
            this.uncachedDecoder = uncachedDecoder;
            return this;
        }

        public Builder setCachedDecoder(String cachedDecoder) {
            this.cachedDecoder = cachedDecoder;
            return this;
        }

        public Builder setMergedDecoder(String mergedDecoder) {
            this.mergedDecoder = mergedDecoder;
            return this;
        }

        public Builder setDecoder(String decoder) {
            this.decoder = decoder;
            return this;
        }
    }


}
