from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from .ingest import ingest
from .build_tokenizer import build_tokenizer
from .stream_tokens import iter_sequences
def main():
 p=argparse.ArgumentParser(description="BELLA.01 27-dataset pipeline");s=p.add_subparsers(dest="cmd",required=True)
 i=s.add_parser("ingest");i.add_argument("--input",type=Path,required=True);i.add_argument("--output",type=Path,required=True)
 t=s.add_parser("tokenizer");t.add_argument("--input",type=Path,required=True);t.add_argument("--output",type=Path,required=True);t.add_argument("--vocab-size",type=int,default=32000)
 x=s.add_parser("stream");x.add_argument("--input",type=Path,required=True);x.add_argument("--tokenizer",type=Path,required=True);x.add_argument("--seq-length",type=int,default=2048);x.add_argument("--stride",type=int);x.add_argument("--include-security",action="store_true");x.add_argument("--limit",type=int,default=0)
 a=p.parse_args()
 if a.cmd=="ingest":print(json.dumps(ingest(a.input,a.output),indent=2))
 elif a.cmd=="tokenizer":print(json.dumps(build_tokenizer(a.input,a.output,a.vocab_size),indent=2))
 else:
  n=0
  for row in iter_sequences(a.input,a.tokenizer,a.seq_length,a.stride,a.include_security):
   print(json.dumps(row,ensure_ascii=False));n+=1
   if a.limit and n>=a.limit:break
  print(json.dumps({"sequences":n}),file=sys.stderr)
if __name__=="__main__":main()
