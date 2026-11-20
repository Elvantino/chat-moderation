# 2-minute demo video script

Record the screen with your voice (OBS, Xbox Game Bar `Win+Alt+R`, or Loom). Only the first 2:00 count.
Upload as **unlisted** on YouTube (or Google Drive with "anyone with the link") and test the link in a
private browser window.

| Time | Show | Say (roughly) |
|---|---|---|
| 0:00–0:15 | Title slide | "Gaming platforms must flag offensive chat on CPU-only servers in under 50 ms. I built a moderation app that compares four models and runs a quantized BERT." |
| 0:15–0:30 | `python app.py` running, browser at 127.0.0.1:7860 | "This is the live app. The primary model is the INT8-quantized BERT; the side panel shows every model's verdict and latency." |
| 0:30–0:45 | Type a **new** clean message, e.g. `nice clutch, gg everyone` | "A clean message is allowed; every model runs in a few milliseconds on CPU." |
| 0:45–1:00 | Type a new obfuscated insult, e.g. `ur such a fcuking 1d10t` | "Leetspeak and a misspelling: the regex normalizer and edit-distance matcher still catch it, and the words are masked." |
| 1:00–1:15 | **Difficult case**, e.g. `that boss fight was sick, I killed it` or an insult without swear words. Switch the dropdown to the TF-IDF baseline and send it again | "A hard case: gaming slang. Here the models disagree. You can switch the deciding model live." |
| 1:15–1:45 | Architecture slide, then the results table slide | "Input goes through regex normalization into five models; the policy masks and labels. On the same test split, INT8 BERT reached ___ macro-F1 at ___ ms, ___× faster than fp32." (read **your** numbers) |
| 1:45–2:00 | Conclusion slide | "Quantization made BERT fit the budget at a small accuracy cost; the next step is distillation, ONNX Runtime and Turkish support." |
