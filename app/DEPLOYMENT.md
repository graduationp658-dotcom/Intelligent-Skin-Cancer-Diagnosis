# Deploying to Hugging Face Spaces

This app (`app/`) is self-contained: model weights, the FAISS knowledge base,
the local Clinical Assistant LLM (Qwen2.5-0.5B-Instruct, quantized to GGUF
q8_0, ~645 MB), and config are all bundled, so the Space needs no external
data downloads or network access at inference time (only the one-time
`all-MiniLM-L6-v2` sentence-transformer download on first build, which is
cached in the image).

## 1. Create the Space

1. Go to https://huggingface.co/new-space.
2. Choose an owner, a name (e.g. `skin-cancer-classifier-demo`), **SDK =
   Gradio**, and **hardware = CPU basic (free)**.
   - The bundled LLM is loaded via `llama-cpp-python` (CPU, GGUF format).
     CPU basic (free, ~16 GB RAM) is comfortably enough to load and run it.
3. Set visibility (public or private) as you prefer.

## 2. Push the `app/` folder as the Space repo

The contents of `app/` should become the *root* of the Space repo (so
`app.py` and `requirements.txt` sit at the top level).

```bash
cd "c:\graduation project\app"
git init
git lfs install
git lfs track "*.pth" "*.bin" "*.gguf"
git add .gitattributes
git add .
git commit -m "Initial Gradio demo: skin lesion classifier + clinical RAG + LLM"

git remote add origin https://huggingface.co/spaces/<your-username>/<space-name>
git push -u origin main
```

`final_model.pth` (~73 MB), `rag/faiss_index.bin`, and
`llm/qwen2.5-0.5b-instruct-q8_0.gguf` (~645 MB) need Git LFS -- the
`git lfs track` step above handles all three. Confirm `.gitattributes` was
created before committing. The total repo size is roughly **~720 MB**, well
within the free-tier 1 GB repo storage quota.

## 3. Build and verify

- The Space will automatically build from `requirements.txt` and launch
  `app.py` (per the `app_file:` field in `README.md`'s metadata header).
- `requirements.txt` includes a second `--extra-index-url` pointing at the
  `llama-cpp-python` CPU wheel index (`abetlen.github.io/llama-cpp-python/whl/cpu`),
  which provides a prebuilt manylinux CPU wheel -- no compiler/build step
  needed on the Space.
- First build takes a few minutes (PyTorch CPU wheel + sentence-transformers
  + llama-cpp-python + downloading `all-MiniLM-L6-v2`, ~80 MB, from the HF
  Hub).
- Once running, upload a test dermoscopy image (e.g. from the ISIC archive)
  and confirm you get a prediction, a Grad-CAM++ overlay, and a report.
- In the Clinical Assistant tab, ask a question and confirm you get a
  grounded, cited answer.

## 4. Notes / things to double check

- **CPU inference speed**: EfficientNet-B4 forward+backward (for Grad-CAM++)
  on CPU takes roughly 1-3 seconds per image on the free tier -- acceptable
  for a demo, but mention this in the UI if it feels slow.
- **Clinical Assistant response time**: the bundled Qwen2.5-0.5B-Instruct
  GGUF (q8_0) model runs via `llama-cpp-python` on CPU and responds in a few
  seconds per turn on the free tier -- much faster than a full-precision 1.5B
  transformers model would be.
- **`sdk_version`**: `app/README.md` pins `sdk_version: 6.17.3` (the version
  this app was tested against). If HF Spaces' supported Gradio versions list
  doesn't include it at deploy time, lower it to the latest available 6.x or
  5.x release -- the `gr.Blocks` / `gr.Image` / `gr.Label` API used here is
  stable across that range.
- **Cold starts**: free CPU Spaces sleep after inactivity; the first request
  after waking will be slow (model + embedder + LLM reload).
- **Disclaimer**: the "not a medical device" banner in `app.py` and the
  `DISCLAIMER` text in every generated report should remain visible -- do not
  remove them if you fork/customize this Space.

## 5. Updating the model later

If you retrain and produce a new `models/checkpoints/final_model.pth`:

```bash
cp "c:\graduation project\models\checkpoints\final_model.pth" "c:\graduation project\app\model\final_model.pth"
cd "c:\graduation project\app"
git add model/final_model.pth
git commit -m "Update model weights"
git push
```

If the clinical knowledge base (Phase 7) is rebuilt, similarly re-copy
`results/rag/{knowledge_base.json,chunk_metadata.json,faiss_index.bin}` into
`app/rag/` and push.

## Alternatives considered

- **Streamlit + Streamlit Community Cloud**: similar free-tier story, but the
  project's interface needs are well served by Gradio's built-in
  `gr.Image`/`gr.Label` components and HF Spaces' tighter integration with
  the HF Hub (model/dataset hosting, git-lfs).
- **Local-only Gradio (`demo.launch(share=True)` or just `localhost`)**: zero
  deployment effort, but not shareable as a persistent link for a graduation
  project writeup/demo -- HF Spaces gives a stable public URL for free.
