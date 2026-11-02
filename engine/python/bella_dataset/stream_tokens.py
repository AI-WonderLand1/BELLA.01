from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Iterator
from tokenizers import Tokenizer
LANES=("architectural_reasoning","full_stack_compilation","agentic_self_learning","secured_sandbox_guard")
LANE_TOKENS={"architectural_reasoning":"[LANE_ARCH]","full_stack_compilation":"[LANE_CODE]","agentic_self_learning":"[LANE_AGENT]","secured_sandbox_guard":"[LANE_SECURITY]"}
def iter_sequences(data:Path,tokenizer_file:Path,seq_length:int=2048,stride:int|None=None,include_security:bool=False)->Iterator[dict]:
    if seq_length<8:raise ValueError("seq_length must be >= 8")
    tok=Tokenizer.from_file(str(tokenizer_file));pad=tok.token_to_id("[PAD]")
    if pad is None:raise ValueError("Tokenizer has no [PAD] token")
    bos=tok.token_to_id("[BOS]");eos=tok.token_to_id("[EOS]");stride=stride or seq_length
    if not 1<=stride<=seq_length:raise ValueError("stride must be between 1 and seq_length")
    for lane in (LANES if include_security else LANES[:-1]):
        file=data/f"{lane}.jsonl"
        if not file.exists():continue
        with file.open(encoding="utf-8") as f:
            for line_no,line in enumerate(f,1):
                if not line.strip():continue
                row=json.loads(line);text=row.get("text")
                if not isinstance(text,str) or not text:continue
                ids=tok.encode(text,add_special_tokens=False).ids
                lane_id=tok.token_to_id(LANE_TOKENS[lane])
                if lane_id is not None:ids.insert(0,lane_id)
                if bos is not None:ids.insert(0,bos)
                if eos is not None:ids.append(eos)
                for start in range(0,len(ids),stride):
                    chunk=ids[start:start+seq_length+1]
                    if len(chunk)<2:continue
                    x=chunk[:-1];y=chunk[1:];mask=[1]*len(x);n=seq_length-len(x)
                    if n>0:x += [pad]*n;y += [-100]*n;mask += [0]*n
                    yield {"input_ids":x[:seq_length],"labels":y[:seq_length],"attention_mask":mask[:seq_length],"lane":lane,"dataset_id":row.get("dataset_id"),"source_file":row.get("source_file"),"line":line_no}
def main():
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--tokenizer",type=Path,required=True);p.add_argument("--seq-length",type=int,default=2048);p.add_argument("--stride",type=int);p.add_argument("--include-security",action="store_true");p.add_argument("--limit",type=int,default=0);a=p.parse_args();n=0
    for row in iter_sequences(a.input,a.tokenizer,a.seq_length,a.stride,a.include_security):
        print(json.dumps(row,ensure_ascii=False));n+=1
        if a.limit and n>=a.limit:break
    print(json.dumps({"sequences":n,"sequence_length":a.seq_length}),file=sys.stderr)
if __name__=="__main__":main()
