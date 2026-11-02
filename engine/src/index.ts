import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { spawn } from "node:child_process";
import { resolve } from "node:path";

const port = Number(process.env.PORT ?? 8787);
const python = process.env.BELLA_PYTHON ?? "python3";
const worker = resolve(process.cwd(), "python/bella_dataset/cli.py");
const inputRoot = process.env.BELLA_DATASET_INPUT;
const outputRoot = process.env.BELLA_DATASET_OUTPUT ?? resolve(process.cwd(), "data/prepared");

function send(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
}

function runPreparation(): Promise<{ stdout: string; stderr: string }> {
  if (!inputRoot) throw new Error("BELLA_DATASET_INPUT is not configured");
  return new Promise((resolvePromise, reject) => {
    const child = spawn(python, [worker, "prepare", "--input", resolve(inputRoot), "--output", resolve(outputRoot)], {
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
    send(res, 200, { status: "ok", service: "bella-hybrid-dataset-engine", version: "0.1.0" });
    return;
  }
  if (req.method === "POST" && req.url === "/api/datasets/prepare") {
    try {
      const result = await runPreparation();
      send(res, 200, { status: "completed", output: outputRoot, ...result });
    } catch (error) {
      send(res, 500, { status: "failed", error: error instanceof Error ? error.message : String(error) });
    }
    return;
  }
  send(res, 404, { error: "Route not found" });
});

server.listen(port, "127.0.0.1", () => {
  console.log(`BELLA.01 dataset engine listening on http://127.0.0.1:${port}`);
});
