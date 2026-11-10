import { AutoTokenizer, env } from "../vendor/transformers/transformers.min.js";

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL("../models/", import.meta.url).href;

let tokenizerPromise;

export function loadTokenizer(onProgress = () => {}) {
  if (!tokenizerPromise) {
    onProgress("Loading local Qwen3 tokenizer");
    tokenizerPromise = AutoTokenizer.from_pretrained("qwen3-4b-tokenizer", {
      local_files_only: true,
      progress_callback: (progress) => {
        const file = progress?.file ? progress.file.split("/").pop() : "tokenizer";
        onProgress("Loading " + file);
      }
    }).catch((error) => {
      tokenizerPromise = null;
      throw new Error("Could not load the bundled Qwen3 tokenizer: " + (error?.message || error));
    });
  }
  return tokenizerPromise;
}

export function tokenizePrompt(tokenizer, prompt) {
  const chatPrompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
  return tokenizer(chatPrompt, {
    padding: "max_length",
    truncation: true,
    max_length: 512,
    return_attention_mask: true
  });
}
