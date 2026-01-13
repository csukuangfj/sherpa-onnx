import { createOnlinePunctuation, onlinePunctuationAddPunct } from "libsherpa_onnx.so";

export class OnlinePunctuationConfig {
  public model: OnlinePunctuationModelConfig = new OnlinePunctuationModelConfig();
}

export class OnlinePunctuationModelConfig {
  public cnnBilstm: string = '';
  public bpeVocab: string = '';
  public numThreads: number = 1;
  public debug: boolean = false;
  public provider: string = 'cpu';
}

export class OnlinePunctuation {
  public handle: object;

  constructor(config: OnlinePunctuationModelConfig) {
    let onlineConfig = new OnlinePunctuationConfig()
    onlineConfig.model = config
    this.handle = createOnlinePunctuation(onlineConfig)
  }

  onlinePunctuationAddPunct(text: string): string {
    let result = onlinePunctuationAddPunct(this.handle, text)
    return result
  }
}
