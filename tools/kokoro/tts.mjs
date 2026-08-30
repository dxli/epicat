// Kokoro TTS worker for epicat.
//
// Reads a JSON job on stdin:
//   {"outDir": "...", "cacheDir": "...", "segments": [{"id":"0001","text":"…",
//    "voice":"af_heart","speed":1.0}, …]}
// Writes <outDir>/<id>.wav for each segment and prints one JSON line per
// segment to stdout: {"id":…, "file":…, "ok":true} or {"id":…, "error":…}.
//
// The model is loaded once and reused for the whole batch.

import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { KokoroTTS } from "kokoro-js";
import { env } from "@huggingface/transformers";

function readStdin() {
  return new Promise((resolve, reject) => {
    let buf = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (d) => (buf += d));
    process.stdin.on("end", () => resolve(buf));
    process.stdin.on("error", reject);
  });
}

const job = JSON.parse(await readStdin());
if (job.cacheDir) env.cacheDir = job.cacheDir;
await mkdir(job.outDir, { recursive: true });

const tts = await KokoroTTS.from_pretrained(
  job.model || "onnx-community/Kokoro-82M-v1.0-ONNX",
  { dtype: job.dtype || "q8", device: "cpu" },
);

for (const seg of job.segments) {
  const file = path.join(job.outDir, `${seg.id}.wav`);
  try {
    const audio = await tts.generate(seg.text, {
      voice: seg.voice || "af_heart",
      speed: seg.speed || 1.0,
    });
    await audio.save(file);
    process.stdout.write(JSON.stringify({ id: seg.id, file, ok: true }) + "\n");
  } catch (err) {
    process.stdout.write(
      JSON.stringify({ id: seg.id, error: String(err && err.message || err) }) + "\n",
    );
  }
}
