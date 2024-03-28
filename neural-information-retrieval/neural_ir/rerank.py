import argparse
from neural_ir.index import Faiss
from neural_ir.models.cross_encoder import CrossEncoder
from neural_ir.utils import write_trec_run
from neural_ir.utils.dataset_utils import read_pairs
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from collections import defaultdict
import ir_measures
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))

def shuffle_words(text, noise):
    words = text.split()
    for i in range(len(words)):
        if torch.rand(1) < noise:
            idx1, idx2 = torch.randint(0, len(words), (2,))
            words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stopwords])

def preprocess_text(text, noise=None, no_stopwords=False):
    if no_stopwords:
        text = remove_stopwords(text)
    if noise > 0.:
        text = shuffle_words(text, noise)
    return text
    

parser = argparse.ArgumentParser(description="Ranking with BiEncoder")
parser.add_argument(
    "--c", type=str, default="data/collection.tsv", help="path to document collection"
)
parser.add_argument(
    "--q", type=str, default="data/test_queries.tsv", help="path to queries"
)
parser.add_argument(
    "--run",
    type=str,
    default="data/test_bm25.trec",
    help="path to the run file of a first-stage ranker (BM25, Dense, Sparse)",
)
parser.add_argument(
    "--device", type=str, default="cuda", help="device to run inference"
)
parser.add_argument("--bs", type=int, default=16, help="batch size")
parser.add_argument(
    "--checkpoint",
    default="output/ce/model",
    type=str,
    help="path to model checkpoint",
)
parser.add_argument(
    "--o",
    type=str,
    default="output/ce/test_run.trec",
    help="path to output run file",
)
parser.add_argument(
    "--noise",
    type=float,
    default=0.0,
    help="randomly swap words in X% of the documents",
)
parser.add_argument(
    "--no-stopwords",
    action="store_true",
    help="remove stopwords from the documents",
)

args = parser.parse_args()

docs = dict(read_pairs(args.c))
queries = dict(read_pairs(args.q))

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
model = CrossEncoder.from_pretrained(args.checkpoint).to(args.device)
model.eval()

pairs_text = []
pairs_id = []
run = ir_measures.read_trec_run(args.run)
count_docs = defaultdict(lambda: 0)
for pair in tqdm(run, desc=f"Reading pairs from {args.run}", position=0, leave=True):
    if count_docs[pair.query_id] < 100:
        pairs_id.append((pair.query_id, pair.doc_id))
        pairs_text.append((queries[pair.query_id], preprocess_text(docs[pair.doc_id], args.noise, args.no_stopwords))
        )
        count_docs[pair.query_id] += 1

results = defaultdict(list)
for idx in range(0, len(pairs_text), args.bs):
    batch_pairs_text = pairs_text[idx : idx + args.bs]
    batch_pairs_id = pairs_id[idx : idx + args.bs]
    batch_inps = tokenizer(
        batch_pairs_text, truncation=True, padding=True, return_tensors="pt"
    ).to(args.device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        scores = model.score_pairs(batch_inps).to("cpu").tolist()
    for pairid, score in zip(batch_pairs_id, scores):
        qid, did = pairid
        results[qid].append((did, score))

output = args.o.split("/")
filename = output[-1]
filename = (
    (f"noise_{100*args.noise}_" if args.noise > 0 else "")
    + ("nostop_" if args.no_stopwords else "")
    + filename
)
output = "/".join(output[:-1] + [filename])
write_trec_run(results, output)
