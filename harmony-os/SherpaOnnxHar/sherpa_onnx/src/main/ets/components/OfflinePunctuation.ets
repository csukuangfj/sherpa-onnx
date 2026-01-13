import { createOfflinePunctuation, offlinePunctuationAddPunct } from "libsherpa_onnx.so";

export class OfflinePunctuationConfig {
  public model: OfflinePunctuationModelConfig = new OfflinePunctuationModelConfig();
}

export class OfflinePunctuationModelConfig {
  public ctTransformer: string = '';
  public numThreads: number = 1;
  public debug: boolean = false;
  public provider: string = 'cpu';
}

export class OfflinePunctuation {
  public handle: object;

  constructor(config: OfflinePunctuationModelConfig) {
    let offlineConfig = new OfflinePunctuationConfig()
    offlineConfig.model = config
    this.handle = createOfflinePunctuation(offlineConfig)
  }

  offlinePunctuationAddPunct(text: string): string {
    let result = offlinePunctuationAddPunct(this.handle, text)
    return result
  }
}