# mojo-bertopic

`mojo-bertopic` is the compute-oriented subset of
[BERTopic](https://github.com/MaartenGr/BERTopic): the c-TF-IDF arithmetic and
the topic-to-topic similarity kernel, moved into one compiled Mojo shared
library. The Python package is named `mojo_bertopic`, so it installs alongside
the real `bertopic` and the tests compare the two directly.

```python
import scipy.sparse as sp
import mojo_bertopic as mbt
from bertopic.vectorizers import CountVectorizer

X = CountVectorizer().fit_transform(["red fox", "blue fox", "green tree", "tall tree"])
mbt.ctfidf(X)                      # same matrix ClassTfidfTransformer().fit_transform gives
mbt.gram(mbt.ctfidf(X))            # topic x topic cosine similarity
mbt.top_n_terms(mbt.ctfidf(X), 3)  # [(column, score), ...] per topic
```

## Why this subset

BERTopic is mostly glue: it calls UMAP, HDBSCAN, a sentence-transformer backend
and Plotly. None of that is arithmetic this port can improve on. What BERTopic
does compute itself, at the top of a c-TF-IDF matrix of `n_topics x vocab`, is:

* `ClassTfidfTransformer.fit` — the document frequency `df`, the truncated mean
  row length, and the idf vector (BM25 variant included);
* `ClassTfidfTransformer.transform` — L1 normalise, optional square root, idf
  scaling, fused into a single pass over the nonzeros;
* `BERTopic._top_n_idx_sparse` / `_top_n_values_sparse` — the per-row top-`n`
  terms behind `get_topics`, `get_topic_info` and every bar chart;
* the topic-to-topic cosine similarity matrix used by `similarity_matrix`,
  `hierarchical_topics` and `find_topics`;
* `representation._mmr.mmr` — the Maximal Marginal Relevance keyword loop.

All five are inner loops over arrays, which is what the port is for.

## Covered subset

| area | implemented API |
| --- | --- |
| c-TF-IDF fit | `fit_idf`, and the `df` / row-length statistics through `col_sums`, `row_sums` |
| c-TF-IDF transform | `ctfidf`, `ctfidf_transform` (`reduce_frequent_words` and the BM25 idf included) |
| Term extraction | `top_n`, `top_n_terms` |
| Topic similarity | `gram`, `rownorm2`, `transpose` |
| Keyword diversity | `mmr` |

### Deliberate differences from upstream

* **No in-place mutation.** `transform` upstream calls
  `normalize(..., copy=False)` and L1-normalises the caller's matrix as a side
  effect. `mojo_bertopic` returns a new matrix and never writes to its input.
  `test_ctfidf_does_not_mutate_its_input` pins both behaviours.
* **Deterministic top-`n`.** Upstream picks columns with `np.argpartition`,
  whose order is undefined, and pads short rows with `None`. This port returns
  columns sorted by descending score, ties broken by the smaller column index,
  short rows padded with `-1`, and the true count per row alongside.
* **`mmr` returns indices**, not words; the caller maps them to its own
  vocabulary. It also caps `top_n` at the number of candidate words instead of
  raising when asked for more.

## Not implemented

Everything else is left to the real `bertopic` on purpose: text
preprocessing and the `CountVectorizer`; UMAP dimensionality reduction; HDBSCAN
clustering; every embedding backend (SentenceTransformer, Cohere, OpenAI,
Gensim, KeyBERT, ...); the Coherence/Positional/TF-IDF/TextGeneration
representation models; `merge_topics`, `reduce_topics`, `reduce_outliers`;
`topics_over_time`, `topics_per_class`, `approximate_distribution`; and the
whole `plotting` subpackage. None of them is an array kernel this port can beat,
and the clustering ones call into C/C++ libraries that are already compiled.

## Install

The repository pins its own Mojo toolchain:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-bertopic.so`. Set `PYTHONPATH=python`
when using the package outside a Pixi task. Using the shared toolchain
directly:

```bash
bash build/build.sh
PYTHONPATH=python python -m pytest tests -q
```

## Performance

Best-of-three wall clock, same process, against the real
`bertopic`/`scikit-learn` code paths. Every case verifies numerical agreement
before timing, so a kernel regression shows up as a failure, not as a fast
number.

| case | reference | mojo-bertopic | result |
| --- | ---: | ---: | ---: |
| c-TF-IDF transform, 1e6 nonzeros | 75.63 ms | 8.34 ms | 9.07x faster |
| top-10 terms, 20000 rows x 60 nnz | 1071.82 ms | 22.78 ms | 47.06x faster |
| topic similarity, 800 x 800, 120 nnz/row | 27.26 ms | 16.84 ms | 1.62x faster |
| ... against a hand-written `X @ X.T` | 1989.69 ms | 16.84 ms | 118x faster |

The top-`n` win is large because upstream runs a Python `for` loop over rows,
calling `np.argpartition` once per row; the Mojo kernel walks the nonzeros
once. The transform win is the fused pass: upstream materialises an
intermediate normalised copy, a sqrt copy and the sparse-diagonal product.
`gram` is a genuinely close call against `sklearn`'s `cosine_similarity`, which
is itself a Cython loop over the same nonzeros — 1.62x is the honest number.
The hand-written dense `X @ X.T` reference is included to show that the
sparse-times-sparse product, not the normalisation, is what the Mojo kernel
actually buys.

Reproduce with:

```bash
pixi run bench
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-bertopic.so`.

The `python/mojo_bertopic` layer owns every array. It normalises any SciPy CSR
input to `Int32` `indptr`/`indices` and `Float64` data (the C ABI cannot
express another dtype), then makes one call per pass. Buffers cross the ABI as
64-bit addresses and are reconstructed in Mojo as
`Pointer[T, AnyOrigin[mut=True]]`, which keeps the exported symbols
non-parametric.

The one structural kernel is `bt_csr_gram`. `A @ A.T` is only cache friendly
when one side is column major, so `bt_csr_transpose` builds the CSC companion
(counting sort over the column indices, `O(nnz + ncols)`) and `bt_csr_gram`
then scatters `a[i,c] * a[j,c]` into row `i` of the output. The row norms come
from `bt_csr_rownorm2` and the normalisation is done in the kernel, so the
whole similarity matrix is one call from Python.

Mojo emits FMA, so results match `bertopic` and `scikit-learn` to a tolerance,
never bit-for-bit. The parity tests use `rtol=1e-12`-ish; exact equality is
asserted only where the operation is genuinely exact (integer counts, index
arrays, the padded top-`n` layout).

## License

MIT
