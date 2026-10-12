# Vendored browser dependencies

Runtime files are copied from the official npm packages without modification.

| Package | Version | License |
| --- | --- | --- |
| onnxruntime-web | 1.20.1 | MIT |
| lottie-web | 5.12.2 | MIT |
| plotly.js-dist-min | 2.35.2 | MIT |
| papaparse | 5.4.1 | MIT |

`ort.min.js`, `ort-wasm-simd-threaded.mjs`, and `ort-wasm-simd-threaded.wasm` must remain on the same version. Inference uses a single WASM thread so the site does not need SharedArrayBuffer or cross-origin isolation headers.
