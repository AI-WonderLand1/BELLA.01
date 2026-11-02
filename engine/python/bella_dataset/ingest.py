from __future__ import annotations
import argparse,bz2,gzip,json,lzma,sqlite3,tarfile,zipfile
from pathlib import Path
from typing import Any
from .common import load_registry,identify_dataset,iter_records,extract_text,digest
SUPPORTED={".json",".jsonl",".ndjson",".csv",".parquet",".pq",".txt",".md",".py",".js",".jsx",".ts",".tsx",".html",".css",".sql",".yaml",".yml",".xml"}
ARCHIVES={".zip",".tar",".tgz",".gz",".bz2",".xz"}
BINARIES={".apk",".dex",".so",".dll",".exe",".bin",".class",".jar",".o",".a"}
def safe_extract(root:Path)->list[str]:
    done=[]
    for archive in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ARCHIVES):
        try:
            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as z:
                    for i in z.infolist():
                        if not (archive.parent/i.filename).resolve().is_relative_to(root.resolve()):raise ValueError(f"Unsafe zip path: {i.filename}")
                        mode=i.external_attr>>16
                        if mode and (mode&0o170000)==0o120000:raise ValueError(f"Symlink rejected: {i.filename}")
                    z.extractall(archive.parent)
            elif tarfile.is_tarfile(archive):
                with tarfile.open(archive,"r:*") as t:
                    for i in t.getmembers():
                        if not (archive.parent/i.name).resolve().is_relative_to(root.resolve()) or not (i.isfile() or i.isdir()):raise ValueError(f"Unsafe tar member: {i.name}")
                    t.extractall(archive.parent)
            elif archive.suffix==".gz":
                with gzip.open(archive,"rb") as src,archive.with_suffix("").open("wb") as dst:
                    while b:=src.read(1024*1024):dst.write(b)
            elif archive.suffix==".bz2":
                with bz2.open(archive,"rb") as src,archive.with_suffix("").open("wb") as dst:
                    while b:=src.read(1024*1024):dst.write(b)
            elif archive.suffix==".xz":
                with lzma.open(archive,"rb") as src,archive.with_suffix("").open("wb") as dst:
                    while b:=src.read(1024*1024):dst.write(b)
            else:continue
            done.append(archive.relative_to(root).as_posix())
        except (OSError,tarfile.TarError,zipfile.BadZipFile) as e:raise RuntimeError(f"Archive extraction failed for {archive}: {e}") from e
    return done
def ingest(source:Path,out:Path)->dict[str,Any]:
    if not source.is_dir():raise FileNotFoundError(source)
    out.mkdir(parents=True,exist_ok=True); reg=load_registry(); archives=safe_extract(source)
    counts={k:0 for k in reg["lanes"]};counts.update(files=0,records_seen=0,records_written=0,duplicates=0,unrecognized=0,file_errors=0,binaries_skipped=0)
    discovered=set();errors=[];dbpath=out/".dedup.sqlite3";dbpath.unlink(missing_ok=True);db=sqlite3.connect(dbpath);db.execute("CREATE TABLE seen (hash TEXT PRIMARY KEY)")
    handles={k:(out/f"{k}.jsonl").open("w",encoding="utf-8") for k in reg["lanes"]}
    try:
        for p in sorted(source.rglob("*")):
            if not p.is_file():continue
            ext=p.suffix.lower()
            if ext in ARCHIVES:continue
            if ext in BINARIES:counts["binaries_skipped"]+=1;continue
            if ext not in SUPPORTED:continue
            rel=p.relative_to(source); ds=next((x for part in rel.parts if (x:=identify_dataset(part,reg))),None)
            if not ds:counts["unrecognized"]+=1;continue
            discovered.add(ds["id"]);lane=ds["lane"];counts["files"]+=1
            try:
                for raw in iter_records(p):
                    counts["records_seen"]+=1;txt=extract_text(raw)
                    if not txt:counts["unrecognized"]+=1;continue
                    h=digest(lane+"\0"+txt)
                    if not db.execute("INSERT OR IGNORE INTO seen(hash) VALUES (?)",(h,)).rowcount:counts["duplicates"]+=1;continue
                    if counts["records_written"]%10000==0:db.commit()
                    handles[lane].write(json.dumps({"text":txt,"lane":lane,"dataset_id":ds["id"],"dataset_name":ds["name"],"source_file":rel.as_posix(),"sha256":h},ensure_ascii=False)+"\n")
                    counts[lane]+=1;counts["records_written"]+=1
            except Exception as e:counts["file_errors"]+=1;errors.append({"file":rel.as_posix(),"error":f"{type(e).__name__}: {e}"})
    finally:
        for f in handles.values():f.close()
        db.commit();db.close();dbpath.unlink(missing_ok=True)
    manifest={"registry_count":len(reg["datasets"]),"datasets_found":sorted(discovered),"datasets_missing":[d["id"] for d in reg["datasets"] if d["id"] not in discovered],"archives_extracted":archives,"counts":counts,"errors":errors,"license_review_required":True,"security_policy":"Executable/binary malware samples are excluded; only text metadata/features enter the security lane."}
    (out/"ingestion_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest
def main():
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(json.dumps(ingest(a.input,a.output),indent=2))
if __name__=="__main__":main()
