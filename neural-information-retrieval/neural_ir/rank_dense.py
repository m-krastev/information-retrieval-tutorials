import argparse
from neural_ir.index import Faiss
from neural_ir.models.dense_encoder import DenseBiEncoder
from neural_ir.utils import write_trec_run
from neural_ir.utils.dataset_utils import read_pairs
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from collections import defaultdict
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
    "--device", type=str, default="cuda", help="device to run inference"
)
parser.add_argument("--bs", type=int, default=16, help="batch size")
parser.add_argument(
    "--checkpoint",
    default="output/dense/model",
    type=str,
    help="path to model checkpoint",
)
parser.add_argument(
    "--o",
    type=str,
    default="output/dense/test_run.trec",
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

docs = read_pairs(args.c)
queries = read_pairs(args.q)

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
model = DenseBiEncoder.from_pretrained(args.checkpoint).to(args.device)
model.eval()
query_embs = []
docs_embs = []
doc_ids = []
for idx in tqdm(
    range(0, len(docs), args.bs), desc="Encoding documents", position=0, leave=True
):
    batch = docs[idx : idx + args.bs]
    docs_texts = [
        preprocess_text(e[1], args.noise, args.no_stopwords) for e in batch
    ]
    doc_ids.extend([e[0] for e in batch])
    docs_inps = tokenizer(
        docs_texts, truncation=True, padding=True, return_tensors="pt"
    ).to(args.device)
    with torch.cuda.amp.autocast(), torch.no_grad():
        batch_embs = model.encode(**docs_inps).to("cpu")
        docs_embs.append(batch_embs)

index = Faiss(d=docs_embs[0].size(1))
docs_embs = torch.cat(docs_embs, dim=0).numpy().astype("float32")
index.add(docs_embs)
# ?for batch_embds in tqdm(docs_embs, desc="Indexing document embeddings"):
# index.add(batch_embs.numpy().astype("float32"))

run = defaultdict(list)
queries_embs = []
for idx in tqdm(
    range(0, len(queries), args.bs),
    desc="Encoding queries and search",
    position=0,
    leave=True,
):
    batch = queries[idx : idx + args.bs]
    query_texts = [e[1] for e in batch]
    query_inps = tokenizer(
        query_texts, truncation=True, padding=True, return_tensors="pt"
    ).to(args.device)
    with torch.cuda.amp.autocast(), torch.no_grad():
        batch_query_embs = (
            model.encode(**query_inps).to("cpu").numpy().astype("float32")
        )
    scores, docs_idx = index.search(batch_query_embs, 1000)
    for idx in range(len(batch)):
        query_id = batch[idx][0]
        for i, score in zip(docs_idx[idx], scores[idx]):
            if i < 0:
                continue
            doc_id = doc_ids[i]
            run[query_id].append((doc_id, score))

output = args.o.split("/")
filename = output[-1]
filename = (f"noise_{100*args.noise}_" if args.noise > 0 else "") + ("nostop_" if args.no_stopwords else "")+ filename
output = "/".join(output[:-1] + [filename])
write_trec_run(run, output)
