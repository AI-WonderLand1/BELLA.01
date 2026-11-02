from __future__ import annotations
import argparse,json
from pathlib import Path
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders,normalizers
def build_tokenizer(source:Path,out:Path,vocab_size:int=32000,min_frequency:int=2)->dict:
    if not source.is_dir():raise FileNotFoundError(source)
    files=sorted(source.glob("*.jsonl"))
    if not files:raise FileNotFoundError("No lane JSONL files found; run ingest first.")
    out.mkdir(parents=True,exist_ok=True)
    tok=Tokenizer(models.BPE(unk_token="[UNK]"));tok.normalizer=normalizers.NFC();tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);tok.decoder=decoders.ByteLevel()
    specials=["[UNK]","[PAD]","[BOS]","[EOS]","[SYSTEM]","[USER]","[ASSISTANT]","[TOOL]","[LANE_ARCH]","[LANE_CODE]","[LANE_AGENT]","[LANE_SECURITY]"]
    trainer=trainers.BpeTrainer(vocab_size=vocab_size,min_frequency=min_frequency,special_tokens=specials,initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    def texts():
        for f in files:
            with f.open(encoding="utf-8") as stream:
                for line in stream:
                    try:r=json.loads(line)
                    except json.JSONDecodeError:continue
                    if isinstance(r.get("text"),str) and r["text"]:yield r["text"]
    tok.train_from_iterator(texts(),trainer=trainer);tok.enable_padding(pad_id=tok.token_to_id("[PAD]"),pad_token="[PAD]")
    path=out/"bella_bpe.json";tok.save(str(path))
    result={"algorithm":"ByteLevel BPE","vocab_size":tok.get_vocab_size(),"requested_vocab_size":vocab_size,"corpus_files":[p.name for p in files],"special_tokens":specials,"tokenizer_file":path.name}
    (out/"tokenizer_manifest.json").write_text(json.dumps(result,indent=2),encoding="utf-8");return result
def main():
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--vocab-size",type=int,default=32000);p.add_argument("--min-frequency",type=int,default=2);a=p.parse_args();print(json.dumps(build_tokenizer(a.input,a.output,a.vocab_size,a.min_frequency),indent=2))
if __name__=="__main__":main()
