// Builds docs/presentation.pptx — results are {{TOKENS}} filled by scripts/fill_slides.py
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require(process.env.SKILL_DIR + "/scripts/apply_theme.js");

const OUT = process.argv[2] || "presentation.pptx";

const HEX = {
  ink: "1B1F2A", slate: "2B3245", white: "FFFFFF", mist: "F1F3F7",
  teal: "2EC4B6", red: "E63946", amber: "F4A261", blue: "4361EE", muted: "8D99AE", grey: "5C6475",
};
const THEME = {
  name: "Moderation Signal", headFontFace: "Arial", bodyFontFace: "Calibri",
  colors: {
    dk1: HEX.ink, lt1: HEX.white, dk2: HEX.slate, lt2: HEX.mist,
    accent1: HEX.teal, accent2: HEX.red, accent3: HEX.amber, accent4: HEX.blue,
    accent5: HEX.muted, accent6: HEX.grey, hlink: HEX.blue, folHlink: HEX.grey,
  },
};

async function icon(Comp, hex, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625
  pres.author = "Matin Jebraeili";
  pres.title = "Real-Time Chat Moderation – SWE015 Use Case 4";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  const FOOTER = "SWE015 · Use Case 4 · Real-Time Chat Moderation · Matin Jebraeili";

  // ---------------------------------------------------------------- layouts
  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: C.text1 },
    objects: [
      { placeholder: { options: { name: "kicker", type: "body", x: 0.6, y: 0.55, w: 8.8, h: 0.35,
        fontSize: 11, bold: true, color: C.accent1, charSpacing: 2, margin: 0 }, text: "" } },
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.05, w: 5.6, h: 1.3,
        fontSize: 38, bold: true, color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "subtitle", type: "body", x: 0.6, y: 2.45, w: 5.4, h: 0.65,
        fontSize: 16, color: C.accent5, valign: "top", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: C.background1 },
    margin: [0.5, 0.5, 0.5, 0.5],
    objects: [
      { placeholder: { options: { name: "kicker", type: "body", x: 0.5, y: 0.28, w: 6, h: 0.28,
        fontSize: 10, bold: true, color: C.accent4, charSpacing: 2, margin: 0 }, text: "" } },
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.56, w: 9, h: 0.55,
        fontSize: 24, bold: true, color: C.text1, align: "left", valign: "top", margin: 0 }, text: "" } },
      { text: { text: FOOTER, options: { x: 0.5, y: 5.27, w: 7.5, h: 0.25, fontSize: 9,
        color: C.accent6, margin: 0 } } },
    ],
    slideNumber: { x: 9.0, y: 5.27, w: 0.5, h: 0.25, fontSize: 9, color: C.accent6, align: "right" },
  });
  pres.defineSlideMaster({
    title: "CLOSING_DARK",
    background: { color: C.text1 },
    objects: [
      { placeholder: { options: { name: "kicker", type: "body", x: 0.5, y: 0.28, w: 6, h: 0.28,
        fontSize: 10, bold: true, color: C.accent1, charSpacing: 2, margin: 0 }, text: "" } },
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.56, w: 9, h: 0.55,
        fontSize: 24, bold: true, color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
      { text: { text: FOOTER, options: { x: 0.5, y: 5.27, w: 7.5, h: 0.25, fontSize: 9,
        color: C.accent5, margin: 0 } } },
    ],
    slideNumber: { x: 9.0, y: 5.27, w: 0.5, h: 0.25, fontSize: 9, color: C.accent5, align: "right" },
  });

  const card = (slide, name, x, y, w, h, fill = C.background2) =>
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { objectName: name, x, y, w, h,
      fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.08 });
  const txt = (slide, name, text, o) => slide.addText(text, { objectName: name, isTextBox: true,
    margin: 0, valign: "top", fontSize: 14, color: C.text1, ...o });
  const iconCircle = async (slide, name, Comp, x, y, d, circle, glyph) => {
    slide.addShape(pres.shapes.OVAL, { objectName: name + " circle", x, y, w: d, h: d,
      fill: { color: circle }, line: { color: circle, width: 0 } });
    const pad = d * 0.24;
    slide.addImage({ objectName: name + " icon", data: await icon(Comp, glyph),
      x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad });
  };
  const arrow = (slide, name, x, y, w, h = 0, color = HEX.muted) =>
    slide.addShape(pres.shapes.LINE, { objectName: name, x, y, w, h,
      line: { color, width: 1.5, endArrowType: "triangle" } });

  // ================================================================ 1 TITLE
  pres.addSection({ title: "Title" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Title" });
  s.addText("SWE015 · INTRODUCTION TO LARGE LANGUAGE MODELS · INDIVIDUAL HOMEWORK", { placeholder: "kicker" });
  s.addText("Real-Time Chat Moderation", { placeholder: "title" });
  s.addText("A four-model ladder for English chat, served by a quantized BERT under a 50 ms CPU budget",
    { placeholder: "subtitle" });
  const info = [
    ["Student", "Matin Jebraeili"],
    ["Student ID", "2309115105"],
    ["Use case", "Use Case 4 – Real-Time Chat Moderation (English)"],
    ["Repository", "{{REPO_URL}}"],
    ["Demo video", "{{VIDEO_URL}}"],
  ];
  info.forEach(([k, v], i) => {
    txt(s, "info label " + i, k, { x: 0.6, y: 3.25 + i * 0.36, w: 1.3, h: 0.3, fontSize: 12,
      bold: true, color: C.accent1 });
    txt(s, "info value " + i, v, { x: 1.9, y: 3.25 + i * 0.36, w: 4.3, h: 0.3, fontSize: 12,
      color: C.background1 });
  });
  // chat-bubble motif: what the app does
  const bubbles = [
    ["gg wp everyone, close match", HEX.teal, "ALLOW"],
    ["you f****** i****", HEX.amber, "MASK"],
    ["[message hidden by moderator]", HEX.red, "BLOCK"],
  ];
  for (let i = 0; i < bubbles.length; i++) {
    const [t, col, tag] = bubbles[i];
    const y = 1.35 + i * 1.05;
    card(s, "bubble " + i, 6.5, y, 2.9, 0.7, HEX.slate);
    txt(s, "bubble text " + i, t, { x: 6.68, y: y + 0.12, w: 2.55, h: 0.3, fontSize: 13, color: C.background1 });
    txt(s, "bubble tag " + i, tag, { x: 6.68, y: y + 0.42, w: 2.55, h: 0.2, fontSize: 9, bold: true,
      color: col, charSpacing: 2 });
  }

  // ================================================================ 2 PROBLEM
  pres.addSection({ title: "Introduction" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("INTRODUCTION", { placeholder: "kicker" });
  s.addText("Gaming chat needs a verdict before the message lands", { placeholder: "title" });
  txt(s, "problem", [
    { text: "The problem", options: { bold: true, fontSize: 16, breakLine: true } },
    { text: "A gaming platform must flag offensive messages in a high-velocity chat stream, mask them, and do it on CPU-only servers within a strict time limit.",
      options: { breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "Who it is for", options: { bold: true, fontSize: 16, breakLine: true } },
    { text: "Players", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – safer chat without waiting for a human", options: { breakLine: true } },
    { text: "Moderators", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – fewer messages to review by hand", options: { breakLine: true } },
    { text: "Platform", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – runs on cheap CPU servers, no GPU at inference" },
  ], { x: 0.5, y: 1.4, w: 4.4, h: 3.6, paraSpaceAfter: 4 });
  txt(s, "success head", "What success looks like", { x: 5.3, y: 1.4, w: 4.2, h: 0.35, fontSize: 16, bold: true });
  const succ = [
    [fa.FaBolt, HEX.teal, "< 50 ms per message", "Quantized BERT meets the latency budget on CPU (mean over ≥ 200 runs)"],
    [fa.FaBalanceScale, HEX.blue, "Macro-F1 on one test set", "Every model scored on the same held-out split; macro-F1 because only a minority is offensive"],
    [fa.FaComments, HEX.amber, "A live app", "Chat UI labels and masks each message, logs latency, runs the INT8 model"],
  ];
  for (let i = 0; i < succ.length; i++) {
    const [Ic, col, h, d] = succ[i];
    const y = 1.85 + i * 1.08;
    card(s, "success card " + i, 5.3, y, 4.2, 0.95);
    await iconCircle(s, "success " + i, Ic, 5.45, y + 0.2, 0.55, col, HEX.white);
    txt(s, "success title " + i, h, { x: 6.15, y: y + 0.1, w: 3.25, h: 0.3, fontSize: 14, bold: true });
    txt(s, "success desc " + i, d, { x: 6.15, y: y + 0.4, w: 3.25, h: 0.5, fontSize: 11, color: C.accent6 });
  }

  // ================================================================ 3 DATA
  pres.addSection({ title: "Development" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · DATA", { placeholder: "kicker" });
  s.addText("Jigsaw comments, one frozen split, one normalizer", { placeholder: "title" });
  card(s, "data card", 0.5, 1.4, 4.3, 3.65);
  await iconCircle(s, "data", fa.FaDatabase, 0.7, 1.55, 0.5, HEX.blue, HEX.white);
  txt(s, "data head", "Jigsaw Toxic Comment Classification", { x: 1.35, y: 1.6, w: 3.35, h: 0.4, fontSize: 14, bold: true });
  txt(s, "data body", [
    { text: "Source: ", options: { bold: true } },
    { text: "Kaggle (Jigsaw / Conversation AI), English Wikipedia talk-page comments", options: { breakLine: true } },
    { text: "Label: ", options: { bold: true } },
    { text: "Offensive = any of toxic, severe_toxic, obscene, threat, insult, identity_hate", options: { breakLine: true } },
    { text: "Size: ", options: { bold: true } },
    { text: "{{N_TOTAL}} comments after de-duplication", options: { breakLine: true } },
    { text: "Splits: ", options: { bold: true } },
    { text: "stratified 80 / 10 / 10, seed 42", options: { breakLine: true } },
    { text: "train {{N_TRAIN}} · val {{N_VAL}} · test {{N_TEST}}", options: { breakLine: true } },
    { text: "Offensive share: ", options: { bold: true } },
    { text: "{{PCT_OFFENSIVE}}" },
  ], { x: 0.75, y: 2.15, w: 3.85, h: 2.8, fontSize: 13, paraSpaceAfter: 6 });

  txt(s, "pre head", "Regex normalization (src/preprocessing.py)", { x: 5.2, y: 1.4, w: 4.3, h: 0.35, fontSize: 14, bold: true });
  const steps = ["Lowercase, strip accents", "Remove URLs, e-mails, IPs, HTML",
    "Leetspeak → letters, only inside words", "Collapse 3+ repeated letters", "Strip punctuation, join “f u c k”"];
  steps.forEach((t, i) => {
    const y = 1.85 + i * 0.42;
    s.addShape(pres.shapes.OVAL, { objectName: "step dot " + i, x: 5.2, y: y + 0.03, w: 0.28, h: 0.28,
      fill: { color: C.accent4 }, line: { color: C.accent4, width: 0 } });
    txt(s, "step num " + i, String(i + 1), { x: 5.2, y: y + 0.03, w: 0.28, h: 0.28, fontSize: 10, bold: true,
      color: C.background1, align: "center", valign: "middle" });
    txt(s, "step " + i, t, { x: 5.6, y: y + 0.02, w: 3.9, h: 0.3, fontSize: 13 });
  });
  card(s, "example card", 5.2, 4.0, 4.3, 1.05, HEX.slate);
  txt(s, "example", [
    { text: "Y0u are a  STUP1D   iiiiidiooot!!!", options: { fontFace: "Courier New", color: C.accent3, breakLine: true } },
    { text: "→  you are a stupid idiot", options: { fontFace: "Courier New", color: C.accent1, bold: true } },
  ], { x: 5.4, y: 4.15, w: 3.95, h: 0.8, fontSize: 12, valign: "middle", paraSpaceAfter: 6 });

  // ================================================================ 4 ARCHITECTURE
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · ARCHITECTURE", { placeholder: "kicker" });
  s.addText("One message in, five models, one moderation decision", { placeholder: "title" });
  const colW = 1.62, gap = 0.27, top = 1.45, H = 3.05;
  const xs = [0.5, 0.5 + (colW + gap), 0.5 + 2 * (colW + gap), 0.5 + 3 * (colW + gap), 0.5 + 4 * (colW + gap)];
  const block = (i, head, sub, body, fill = C.background2, headCol = C.text1) => {
    card(s, "arch col " + i, xs[i], top, colW, H, fill);
    txt(s, "arch head " + i, head, { x: xs[i] + 0.1, y: top + 0.12, w: colW - 0.2, h: 0.5, fontSize: 12, bold: true, color: headCol });
    txt(s, "arch sub " + i, sub, { x: xs[i] + 0.1, y: top + 0.62, w: colW - 0.2, h: 0.3, fontSize: 9, color: C.accent6 });
    if (body) txt(s, "arch body " + i, body, { x: xs[i] + 0.1, y: top + 0.98, w: colW - 0.2, h: H - 1.1, fontSize: 10.5, paraSpaceAfter: 4 });
  };
  block(0, "User input", "app.py · Gradio · browser", [
    { text: "Typed message", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "“Stream test messages” replay", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "Primary-model dropdown", options: { bullet: { indent: 12 } } }]);
  block(1, "Preprocessing", "src/preprocessing.py", [
    { text: "normalize_text(): regex leetspeak, repeats, URLs, punctuation", options: { breakLine: true } },
    { text: "Applied inside every model; BERT then uses its WordPiece tokenizer" }]);
  card(s, "arch col 2", xs[2], top, colW, H, C.background2);
  txt(s, "arch head 2", "Models (all on CPU)", { x: xs[2] + 0.1, y: top + 0.12, w: colW - 0.2, h: 0.3, fontSize: 12, bold: true });
  const mods = [["Lexicon rule", "rules · OSA ≤ 1"], ["TF-IDF + LogReg", "fp64 · sklearn"], ["Word2Vec + FFNN", "fp32 · PyTorch"],
    ["BERT-base", "fp32 · Transformers"], ["BERT-base ★", "dynamic INT8 · primary"]];
  mods.forEach(([m, p], i) => {
    const y = top + 0.5 + i * 0.5;
    const isQ = i === 4;
    card(s, "model box " + i, xs[2] + 0.08, y, colW - 0.16, 0.44, isQ ? C.accent1 : C.background1);
    txt(s, "model name " + i, m, { x: xs[2] + 0.16, y: y + 0.03, w: colW - 0.3, h: 0.2, fontSize: 10, bold: true });
    txt(s, "model prec " + i, p, { x: xs[2] + 0.16, y: y + 0.22, w: colW - 0.3, h: 0.18, fontSize: 8.5, color: isQ ? C.text1 : C.accent6 });
  });
  block(3, "Post-processing", "src/moderation.py", [
    { text: "Primary model → ALLOW / MASK / FLAG / BLOCK", options: { breakLine: true } },
    { text: "LexiconMatcher masks offensive words: “f******”" }]);
  block(4, "Output", "Gradio chat + side panel", [
    { text: "Masked message + verdict, p(offensive)", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "Per-model latency vs 50 ms", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "Session log + benchmark table", options: { bullet: { indent: 12 } } }]);
  for (let i = 0; i < 4; i++) arrow(s, "arrow " + i, xs[i] + colW + 0.03, top + H / 2, gap - 0.06);
  card(s, "runs on", 0.5, 4.62, 9.0, 0.48, HEX.slate);
  txt(s, "runs on text", [
    { text: "Where it runs: ", options: { bold: true, color: C.accent1 } },
    { text: "everything on CPU – measured runs: cloud VM, Intel Xeon 2.3 GHz, 2 vCPUs · no GPU used · app: local Python process · weights: Hugging Face Hub → models/",
      options: { color: C.background1 } }],
    { x: 0.65, y: 4.66, w: 8.75, h: 0.4, fontSize: 10.5, valign: "middle" });

  // ================================================================ 5 TECHNIQUE MAP
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · TECHNIQUE MAP", { placeholder: "kicker" });
  s.addText("Technique map: every area has a working component", { placeholder: "title" });
  const areaCell = (label, color) => ({ text: label, options: { bold: true, color: C.background1, fill: { color } } });
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 } } });
  const rows = [
    [hdr("Area"), hdr("Technique"), hdr("Component in the application (file)")],
    [areaCell("A · Neural networks", HEX.blue), "Feedforward net, SGD + backprop, dropout", "Word2Vec + FFNN classifier (neural_net.py)"],
    [areaCell("B · Classical NLP", HEX.teal), "Regex normalization (leetspeak, repeated letters)", "Input of every model (preprocessing.py)"],
    [areaCell("B · Classical NLP", HEX.teal), "Minimum edit distance (OSA) lexicon matcher", "Rule baseline + masking (rule_baseline.py)"],
    [areaCell("B · Classical NLP", HEX.teal), "tf-idf (word + char n-grams) + logistic regression", "Classical baseline (baseline.py)"],
    [areaCell("B · Classical NLP", HEX.teal), "word2vec trained on the training messages", "FFNN embedding layer (neural_net.py)"],
    [areaCell("C · Transformers", HEX.amber), "BERT fine-tuning with a task head; WordPiece", "bert-base-uncased (transformer_model.py)"],
    [areaCell("D · Efficient models", HEX.red), "Dynamic INT8 quantization; size/latency/F1 benchmark", "Primary app model (quantize.py, evaluate.py)"],
  ];
  s.addTable(rows, { objectName: "technique table", x: 0.5, y: 1.35, w: 9.0, colW: [1.75, 3.75, 3.5],
    fontSize: 12, color: C.text1, fontFace: "Calibri", rowH: 0.38, valign: "middle",
    border: { type: "solid", pt: 0.75, color: "D5D9E2" }, margin: [2, 5, 2, 5] });
  txt(s, "swap note", [
    { text: "Component swap (allowed within an area): ", options: { bold: true } },
    { text: "the brief suggests BERTurk or multilingual BERT; I use English bert-base-uncased because the data (Jigsaw) is English." }],
    { x: 0.5, y: 4.6, w: 9.0, h: 0.45, fontSize: 11, color: C.accent6 });

  // ================================================================ 6 MODELS
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · MODELS", { placeholder: "kicker" });
  s.addText("The ladder, and how each model was trained", { placeholder: "title" });
  const mcards = [
    [fa.FaSpellCheck, HEX.teal, "Lexicon rule", "No training. Seed list + 88 terms mined from the train split (TF-IDF/LR weight, ≥ 60% offensive); offensive if any token matches, incl. OSA distance 1."],
    [fa.FaChartBar, HEX.teal, "TF-IDF + LogReg", "Word 1–2-grams + char_wb 2–5-grams, sublinear tf. Logistic regression (liblinear, C = 4, class-balanced)."],
    [fa.FaProjectDiagram, HEX.blue, "Word2Vec + FFNN", "Skip-gram 200-d, window 5, 10 epochs. Mean-pooled embeddings → LayerNorm → 256 → 64 → 1. SGD (momentum 0.9), lr 0.05, early stopping."],
    [fa.FaBrain, HEX.amber, "BERT-base (fp32)", "bert-base-uncased, 128 tokens. HF Trainer on CPU: 6,000-message stratified train subset, 1 epoch, AdamW 2e-5, batch 16 × 2, class-weighted loss."],
    [fa.FaCompressArrowsAlt, HEX.red, "BERT-base (INT8)", "Fine-tuned BERT after post-training dynamic INT8 quantization (next slide). No retraining."],
    [fa.FaClipboardCheck, HEX.grey, "Shared protocol", "Same full test split (15,958) for all. Threshold tuned on validation for macro-F1 (BERT: 4,000-message val subset)."],
  ];
  for (let i = 0; i < mcards.length; i++) {
    const [Ic, col, h, d] = mcards[i];
    const cx = 0.5 + (i % 3) * 3.07, cy = 1.4 + Math.floor(i / 3) * 1.85;
    card(s, "model card " + i, cx, cy, 2.86, 1.7);
    await iconCircle(s, "model " + i, Ic, cx + 0.15, cy + 0.15, 0.45, col, HEX.white);
    txt(s, "model card title " + i, h, { x: cx + 0.7, y: cy + 0.22, w: 2.05, h: 0.32, fontSize: 14, bold: true });
    txt(s, "model card body " + i, d, { x: cx + 0.15, y: cy + 0.7, w: 2.6, h: 0.95, fontSize: 10.5, color: C.text2 });
  }

  // ================================================================ 7 QUANTIZATION
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · QUANTIZATION", { placeholder: "kicker" });
  s.addText("Dynamic INT8: int8 Linear layers, fp32 embeddings", { placeholder: "title" });
  txt(s, "quant method", [
    { text: "torch.ao.quantization.quantize_dynamic(",
      options: { fontFace: "Courier New", fontSize: 11, color: C.accent4, breakLine: true } },
    { text: "    model, {nn.Linear}, dtype=torch.qint8)",
      options: { fontFace: "Courier New", fontSize: 11, color: C.accent4, breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Weights of every Linear layer (attention Q/K/V/output, feed-forward, classifier head) stored as int8", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "Activations quantized on the fly per input; int8 matmuls on x86 (fbgemm) kernels", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "Embeddings and LayerNorm stay fp32, so the file shrinks by less than 4×", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "Decision threshold re-tuned on validation after quantization", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "Saved as an int8 state_dict and reloaded into a quantized skeleton: the app loads exactly what was evaluated", options: { bullet: { indent: 12 } } },
  ], { x: 0.5, y: 1.4, w: 4.6, h: 3.65, fontSize: 13, paraSpaceAfter: 6 });
  const stats = [
    ["Size (MB)", "{{bert_fp32.SIZE}}", "{{bert_int8.SIZE}}"],
    ["Latency (ms, CPU)", "{{bert_fp32.LAT}}", "{{bert_int8.LAT}}"],
    ["Macro-F1", "{{bert_fp32.F1}}", "{{bert_int8.F1}}"],
  ];
  txt(s, "fp32 label", "fp32", { x: 6.85, y: 1.4, w: 1.2, h: 0.3, fontSize: 11, bold: true, color: C.accent6, align: "center" });
  txt(s, "int8 label", "INT8", { x: 8.25, y: 1.4, w: 1.2, h: 0.3, fontSize: 11, bold: true, color: C.accent1, align: "center" });
  stats.forEach(([k, a, b], i) => {
    const y = 1.75 + i * 1.1;
    card(s, "stat card " + i, 5.45, y, 4.05, 0.95);
    txt(s, "stat label " + i, k, { x: 5.6, y: y + 0.1, w: 1.25, h: 0.75, fontSize: 12, bold: true, valign: "middle" });
    txt(s, "stat fp32 " + i, a, { x: 6.85, y: y + 0.1, w: 1.2, h: 0.75, fontSize: 22, color: C.accent6, align: "center", valign: "middle" });
    txt(s, "stat arrow " + i, "→", { x: 8.0, y: y + 0.1, w: 0.3, h: 0.75, fontSize: 18, color: C.accent5, align: "center", valign: "middle" });
    txt(s, "stat int8 " + i, b, { x: 8.25, y: y + 0.1, w: 1.2, h: 0.75, fontSize: 22, bold: true, color: C.accent1, align: "center", valign: "middle" });
  });

  // ================================================================ 8 RESULTS TABLE
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · RESULTS", { placeholder: "kicker" });
  s.addText("Results on the same held-out test split", { placeholder: "title" });
  const H2 = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, align: "center" } });
  const R = (k, name, area, prec, hl = false) => {
    const o = hl ? { fill: { color: "D5F5F2" }, bold: true } : {};
    const c = (t, align = "center") => ({ text: t, options: { align, ...o } });
    return [c(name, "left"), c(area, "left"), c(prec), c(`{{${k}.F1}}`), c(`{{${k}.SIZE}}`), c(`{{${k}.LAT}}`), c(`{{${k}.P95}}`), c(`{{${k}.OK}}`)];
  };
  s.addTable([
    [H2("Model"), H2("Area"), H2("Precision"), H2("Macro-F1"), H2("Size (MB)"), H2("Avg latency (ms)"), H2("p95 (ms)"), H2("< 50 ms")],
    R("lexicon_rule", "Lexicon rule (edit distance)", "B", "rules"),
    R("baseline", "TF-IDF + LogReg", "B", "fp64"),
    R("neural_net", "Word2Vec + FFNN (SGD)", "A", "fp32"),
    R("bert_fp32", "BERT-base-uncased", "C", "fp32"),
    R("bert_int8", "BERT-base-uncased ★ (app)", "D", "INT8", true),
  ], { objectName: "results table", x: 0.5, y: 1.4, w: 9.0, colW: [2.35, 0.6, 0.85, 1.0, 1.0, 1.35, 0.95, 0.9],
    fontSize: 12, color: C.text1, fontFace: "Calibri", rowH: 0.46, valign: "middle",
    border: { type: "solid", pt: 0.75, color: "D5D9E2" }, margin: [2, 5, 2, 5] });
  card(s, "protocol card", 0.5, 4.3, 9.0, 0.78);
  txt(s, "protocol", [
    { text: "Protocol: ", options: { bold: true } },
    { text: "macro-F1 on all {{N_TEST}} test messages · latency = mean of {{RUNS}} single-message calls (batch size 1) after warm-up, CPU only, same messages for every model", options: { breakLine: true } },
    { text: "Hardware: ", options: { bold: true } },
    { text: "{{HARDWARE}}" }],
    { x: 0.65, y: 4.37, w: 8.7, h: 0.65, fontSize: 11, valign: "middle", paraSpaceAfter: 2 });

  // ================================================================ 9 COMPARISON
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
  s.addText("CONCLUSION", { placeholder: "kicker" });
  s.addText("What the comparison showed", { placeholder: "title" });
  const labels = ["Lexicon rule", "TF-IDF + LR", "W2V + FFNN", "BERT fp32", "BERT INT8"];
  const chartBase = (title, color) => ({
    barDir: "bar", showTitle: true, title, titleFontSize: 12, titleColor: HEX.ink, titleFontFace: "+mj-lt",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9, dataLabelColor: HEX.slate, dataLabelFontFace: "+mn-lt",
    chartColors: [color], catAxisLabelColor: HEX.slate, catAxisLabelFontSize: 9, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelColor: HEX.grey, valAxisLabelFontSize: 8, valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: "E3E6ED", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    catAxisOrientation: "maxMin", valAxisMinVal: 0,
  });
  s.addChart(pres.charts.BAR, [{ name: "Macro-F1", labels, values: [0, 0, 0, 0, 0] }],
    { objectName: "chart macro-f1", x: 0.5, y: 1.3, w: 4.4, h: 2.45, ...chartBase("Macro-F1 (test split)", HEX.blue), valAxisMaxVal: 1, dataLabelFormatCode: "0.000", valAxisLabelFormatCode: "0.0" });
  s.addChart(pres.charts.BAR, [{ name: "Latency (ms)", labels, values: [0, 0, 0, 0, 0] }],
    { objectName: "chart latency", x: 5.1, y: 1.3, w: 4.4, h: 2.45, ...chartBase("Mean CPU latency per message (ms)", HEX.teal), dataLabelFormatCode: "0.0", valAxisLabelFormatCode: "0" });
  const takeaways = [
    [fa.FaTrophy, HEX.blue, "Most accurate", "{{BEST_F1_MODEL}}", "macro-F1 {{BEST_F1}}"],
    [fa.FaTachometerAlt, HEX.teal, "Fastest", "{{FASTEST_MODEL}}", "{{FASTEST_LAT}} ms per message"],
    [fa.FaCompressArrowsAlt, HEX.red, "INT8 vs fp32 BERT", "{{INT8_SPEEDUP}}× faster", "{{INT8_SIZE_RATIO}}× smaller · ΔF1 {{INT8_F1_DELTA}}"],
    [fa.FaStopwatch, HEX.amber, "Latency budget", "{{N_WITHIN}} of {{N_MODELS}} models", "under 50 ms on CPU"],
  ];
  for (let i = 0; i < takeaways.length; i++) {
    const [Ic, col, k, v, d] = takeaways[i];
    const cx = 0.5 + i * 2.29;
    card(s, "takeaway " + i, cx, 3.9, 2.14, 1.18);
    await iconCircle(s, "takeaway " + i, Ic, cx + 0.12, cy0(3.9), 0.36, col, HEX.white);
    txt(s, "takeaway key " + i, k, { x: cx + 0.56, y: 3.99, w: 1.5, h: 0.25, fontSize: 10, bold: true, color: C.accent6 });
    txt(s, "takeaway val " + i, v, { x: cx + 0.12, y: 4.35, w: 1.95, h: 0.32, fontSize: 13, bold: true });
    txt(s, "takeaway desc " + i, d, { x: cx + 0.12, y: 4.67, w: 1.95, h: 0.35, fontSize: 10, color: C.text2 });
  }
  function cy0(y) { return y + 0.08; }

  // ================================================================ 10 LIMITATIONS
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
  s.addText("CONCLUSION · LIMITATIONS", { placeholder: "kicker" });
  s.addText("Limitations, with a real failure from the test split", { placeholder: "title" });
  card(s, "failure card", 0.5, 1.4, 4.3, 3.65, HEX.slate);
  await iconCircle(s, "failure", fa.FaExclamationTriangle, 0.7, 1.55, 0.45, HEX.red, HEX.white);
  txt(s, "failure head", "Failure example · {{FAIL_MODEL}}", { x: 1.3, y: 1.6, w: 3.4, h: 0.4, fontSize: 13, bold: true, color: C.background1 });
  txt(s, "failure text", "“{{FAIL_TEXT}}”", { x: 0.7, y: 2.15, w: 3.9, h: 1.15, fontSize: 13, italic: true, color: C.background1 });
  txt(s, "failure verdict", "True: {{FAIL_TRUE}} · Predicted: {{FAIL_PRED}} (p = {{FAIL_PROB}})",
    { x: 0.7, y: 3.35, w: 3.9, h: 0.3, fontSize: 11, bold: true, color: C.accent3 });
  txt(s, "failure why", [
    { text: "Why: ", options: { bold: true, color: C.accent1 } },
    { text: "{{FAIL_WHY}}", options: { color: C.background1 } }],
    { x: 0.7, y: 3.75, w: 3.9, h: 1.2, fontSize: 11 });
  txt(s, "limits", [
    { text: "Domain shift", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – Wikipedia talk comments, not game chat (“killed it”, “trash”)", options: { breakLine: true } },
    { text: "English only", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – Turkish needs BERTurk and OffensEval-TR", options: { breakLine: true } },
    { text: "CPU-only training", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – BERT saw 6k messages (1 epoch) vs 128k for TF-IDF", options: { breakLine: true } },
    { text: "One message at a time", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – no conversation context or user history", options: { breakLine: true } },
    { text: "Masking needs the lexicon", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – implicit toxicity is flagged but has no word to mask", options: { breakLine: true } },
    { text: "Dataset bias", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – identity terms are over-represented in toxic Jigsaw comments" },
  ], { x: 5.15, y: 1.4, w: 4.35, h: 3.65, fontSize: 12.5, paraSpaceAfter: 7 });

  // ================================================================ 11 NEXT
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
  s.addText("CONCLUSION · NEXT STEPS", { placeholder: "kicker" });
  s.addText("What I would do next", { placeholder: "title" });
  const nexts = [
    [fa.FaFeatherAlt, HEX.red, "Distil + ONNX Runtime", "Distil BERT into a smaller student (e.g. DistilBERT), export to ONNX Runtime with INT8 for more latency headroom."],
    [fa.FaLanguage, HEX.blue, "Turkish support", "BERTurk on OffensEval-TR 2020; add Turkish-character rules (ı/i, ş/s) to the normalizer."],
    [fa.FaHistory, HEX.amber, "Conversation context", "Feed the previous messages and the user's recent history so sarcasm and targeted harassment become visible."],
    [fa.FaUserCheck, HEX.teal, "Moderator feedback loop", "Log moderator overrides from the app, retrain on them, and split the label into insult / threat / hate."],
  ];
  for (let i = 0; i < nexts.length; i++) {
    const [Ic, col, h, d] = nexts[i];
    const cx = 0.5 + (i % 2) * 4.6, cy = 1.4 + Math.floor(i / 2) * 1.85;
    card(s, "next card " + i, cx, cy, 4.4, 1.65);
    await iconCircle(s, "next " + i, Ic, cx + 0.2, cy + 0.25, 0.6, col, HEX.white);
    txt(s, "next title " + i, h, { x: cx + 1.0, y: cy + 0.25, w: 3.25, h: 0.35, fontSize: 15, bold: true });
    txt(s, "next body " + i, d, { x: cx + 1.0, y: cy + 0.65, w: 3.25, h: 0.9, fontSize: 12, color: C.text2 });
  }

  // ================================================================ 12 AI USAGE
  s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Conclusion" });
  s.addText("CONCLUSION · AI USAGE STATEMENT", { placeholder: "kicker" });
  s.addText("How AI tools were used in this project", { placeholder: "title" });
  const ai = [
    [fa.FaRobot, HEX.blue, "Tool", "Claude (Anthropic), used through claude.ai."],
    [fa.FaCode, HEX.amber, "Used for", "Drafting the code of every module (preprocessing, the four models, quantization, evaluation, Gradio app), the unit tests, the README and the first draft of these slides."],
    [fa.FaUserCheck, HEX.teal, "My own work", "Claude also executed the training, evaluation and latency runs in a cloud sandbox (Intel Xeon, 2 vCPUs) at my request. I reviewed the code and results and can explain every component."],
    [fa.FaCheckCircle, HEX.red, "Numbers", "Every number was measured by running this project's code and inserted from models/results/ by scripts/fill_slides.py. None comes from an AI answer, paper or model card."],
  ];
  for (let i = 0; i < ai.length; i++) {
    const [Ic, col, h, d] = ai[i];
    const y = 1.4 + i * 0.93;
    card(s, "ai card " + i, 0.5, y, 9.0, 0.8, HEX.slate);
    await iconCircle(s, "ai " + i, Ic, 0.65, y + 0.14, 0.52, col, HEX.white);
    txt(s, "ai head " + i, h, { x: 1.35, y: y + 0.12, w: 1.6, h: 0.55, fontSize: 14, bold: true, color: C.background1, valign: "middle" });
    txt(s, "ai body " + i, d, { x: 3.0, y: y + 0.1, w: 6.35, h: 0.6, fontSize: 12, color: C.background1, valign: "middle" });
  }
  s.addNotes("Edit this slide if you used any other AI tool (e.g. Copilot, ChatGPT) or used Claude for other parts.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote " + OUT);
})();
