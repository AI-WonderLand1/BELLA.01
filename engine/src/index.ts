import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { spawn } from "node:child_process";
import { resolve } from "node:path";

const port = Number(process.env.PORT ?? 8787);
const python = process.env.BELLA_PYTHON ?? "python3";
const worker = resolve(process.cwd(), "python/bella_dataset/cli.py");
const inputRoot = process.env.BELLA_DATASET_INPUT;
const outputRoot = process.env.BELLA_DATASET_OUTPUT ?? resolve(process.cwd(), "data/prepared");
const kaggleDownloadRoot = process.env.BELLA_KAGGLE_DOWNLOAD_DIR ?? resolve(process.cwd(), "data/kaggle");

function send(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
}

function runWorker(args: string[]): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(python, [worker, ...args], {
      stdio: ["ignore", "pipe", "pipe"],
      env: process.env
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8").on("data", (chunk: string) => { stdout += chunk; });
    child.stderr.setEncoding("utf8").on("data", (chunk: string) => { stderr += chunk; });
    child.once("error", reject);
    child.once("close", (code) => {
      if (code === 0) resolvePromise({ stdout, stderr });
      else reject(new Error(`Python worker exited with code ${code}: ${stderr || stdout}`));
    });
  });
}

const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
  if (req.method === "GET" && req.url === "/health") {
    send(res, 200, { status: "ok", service: "bella-hybrid-dataset-engine", version: "0.2.0" });
    return;
  }
  try {
    if (req.method === "POST" && req.url === "/api/datasets/download-kaggle") {
      const dataset = process.env.KAGGLE_DATASET;
      if (!dataset) {
        send(res, 400, { status: "failed", error: "Set KAGGLE_DATASET=owner/dataset-slug" });
        return;
      }
      const result = await runWorker(["download-kaggle", "--dataset", dataset, "--output", resolve(kaggleDownloadRoot)]);
      send(res, 200, { status: "completed", provider: "kaggle", destination: kaggleDownloadRoot, ...result });
      return;
    }
    if (req.method === "POST" && req.url === "/api/datasets/prepare") {
      if (!inputRoot) {
        send(res, 400, { status: "failed", error: "BELLA_DATASET_INPUT is not configured" });
        return;
      }
      const result = await runWorker(["prepare", "--input", resolve(inputRoot), "--output", resolve(outputRoot)]);
      send(res, 200, { status: "completed", output: outputRoot, ...result });
      return;
    }
    if (req.method === "POST" && req.url === "/api/datasets/mirror-huggingface") {
      const result = await runWorker(["mirror-huggingface", "--input", resolve(outputRoot)]);
      send(res, 200, { status: "completed", provider: "huggingface", source: outputRoot, ...result });
      return;
    }
  } catch (error) {
    send(res, 500, { status: "failed", error: error instanceof Error ? error.message : String(error) });
    return;
  }
  send(res, 404, { error: "Route not found" });
});

server.listen(port, "127.0.0.1", () => {
  console.log(`BELLA.01 dataset engine listening on http://127.0.0.1:${port}`);
});
