from __future__ import annotations
import csv, hashlib, json, re
from pathlib import Path
from typing import Any, Iterator
REGISTRY_PATH=Path(__file__).with_name("dataset_registry.json")
def load_registry()->dict[str,Any]: return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
def identify_dataset(name:str,registry:dict[str,Any]|None=None)->dict[str,Any]|None:
    key=re.sub(r"[^a-z0-9]+","",name.lower())
    for item in (registry or load_registry())["datasets"]:
        aliases={re.sub(r"[^a-z0-9]+","",item["id"].lower()),re.sub(r"[^a-z0-9]+","",item["name"].lower())}
        if key in aliases or any(a and (a in key or key in a) for a in aliases): return item
    return None
def digest(value:str)->str: return hashlib.sha256(value.encode("utf-8")).hexdigest()
def iter_records(path:Path)->Iterator[Any]:
    s=path.suffix.lower()
    if s in {".jsonl",".ndjson"}:
        with path.open(encoding="utf-8",errors="replace") as f:
            for n,line in enumerate(f,1):
                if line.strip():
                    try: yield json.loads(line)
                    except json.JSONDecodeError as e: raise ValueError(f"Invalid JSONL line {n}: {e}") from e
    elif s==".json":
        with path.open(encoding="utf-8",errors="replace") as f:
            first=""
            while not first:
                ch=f.read(1)
                if not ch: return
                if not ch.isspace(): first=ch
            f.seek(0)
            if first=="[":
                try: import ijson
                except ImportError as e: raise RuntimeError("Large JSON arrays require ijson; install requirements-pipeline.txt") from e
                yield from ijson.items(f,"item")
            else:
                data=json.load(f)
                if isinstance(data,dict):
                    for k in ("data","records","examples","train","validation","test"):
                        if isinstance(data.get(k),list): yield from data[k]; return
                yield data
    elif s==".csv":
        with path.open(encoding="utf-8",errors="replace",newline="") as f: yield from csv.DictReader(f)
    elif s in {".parquet",".pq"}:
        try: import pyarrow.parquet as pq
        except ImportError as e: raise RuntimeError("Parquet requires pyarrow") from e
        pf=pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1024): yield from batch.to_pylist()
    else:
        with path.open(encoding="utf-8",errors="replace") as f:
            for line in f:
                if line.strip(): yield {"text":line.rstrip("\n")}
TEXT_KEYS=("role","from","speaker","text","content","code","source","target","instruction","prompt","question","answer","response","completion","output","input","documentation","description","diff","patch","commit_message","thought","reasoning","trajectory","trace","tool_call","tool_output","steps","messages","conversations","chosen","rejected")
def _flatten(value:Any,depth:int=0)->list[str]:
    if depth>10 or value is None:return []
    if isinstance(value,str):return [value.strip()] if value.strip() else []
    if isinstance(value,(int,float,bool)):return [str(value)]
    if isinstance(value,list):
        out=[]
        for item in value:out.extend(_flatten(item,depth+1))
        return out
    if isinstance(value,dict):
        out=[]
        ordered=[k for k in TEXT_KEYS if k in value]
        ordered.extend(k for k in sorted(value) if k not in ordered)
        for key in ordered:
            parts=_flatten(value[key],depth+1)
            if parts:out.append(f"<{key}> "+" ".join(parts))
        return out
    return []
def extract_text(record:Any)->str:
    if isinstance(record,str):return record.strip()[:2_000_000]
    return "\n\n".join(x for x in _flatten(record) if x)[:2_000_000]
