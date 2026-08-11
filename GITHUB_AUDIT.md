# GitHub audit — consolidated findings and action plan

Everything found across the review of `github.com/thakerpragnesh`, in one place.
All figures below were verified by cloning the repositories and running the code,
not by reading file listings. Original audit: 10 August 2026. Addendum from a
second, independent clone-and-verify pass: 11 August 2026 (section 10).

**Contents**
1. [Executive summary](#1-executive-summary)
2. [Profile-level findings](#2-profile-level-findings)
3. [Repository inventory](#3-repository-inventory)
4. [Code audit: the Max-3 provenance problem](#4-code-audit-the-max-3-provenance-problem)
5. [Code audit: confirmed defects](#5-code-audit-confirmed-defects)
6. [Hygiene and privacy](#6-hygiene-and-privacy)
7. [The replacement: structured-pruning](#7-the-replacement-structured-pruning)
8. [Action plan](#8-action-plan)
9. [Open questions](#9-open-questions)
10. [Addendum — second independent verification pass](#10-addendum-second-independent-verification-pass)

---

## 1. Executive summary

Five findings, in order of how much they matter.

**The GitHub link on the résumé currently works against you.** A recruiter who
clicks sees a bio saying "pursuing Ph.D" (completed 2025, with "Surathkal"
misspelled), a pinned fork of someone else's textbook, and a top repository with
no README whose files are named `12 VGGNetPruning.ipynb` through `16.ipynb`.
Nothing visible connects that page to two IEEE Access journal articles.

**No public repository contains a working Max-3 implementation.** Versions 1–3 of
the pruning framework are pure L1 norm with no Max-3 code at all. Version 4 adds
Max-3 but wires it wrong — it computes the score and then selects channels using a
separate L1 tensor. Your headline published contribution has no correct public
implementation.

**Eleven repositories hold what is really one project.** Four `pruning_framwork`
repos version by repository name rather than branch. Five more are completely
empty. `thakerpragnesh` — the repo GitHub renders as your profile page — is one of
the empty ones.

**A privacy issue needs fixing regardless of everything else.** `DeepLearning/Major
Project/` contains four directories named after individual students (and, per the
addendum, two files of actual student feedback — worse than originally scoped).

**The underlying research is genuinely strong.** Six peer-reviewed publications
including two IEEE Access articles. The problem is entirely presentation and
packaging, not substance.

---

## 2. Profile-level findings

| Field | Current | Problem |
|---|---|---|
| Bio | "I am pursuing Ph.D in Information Technology from NITK Suthakal and working on Deep Learning" | Ph.D. completed 2025; "Surathkal" misspelled |
| Location | Rajkot | Résumé says Ahmedabad |
| Website | nitk.ac.in | Should be LinkedIn |
| Pinned repo #2 | `neural-networks-and-deep-learning` | A **fork of Michael Nielsen's book code** — not your work, occupying half your visible profile |
| Profile README | none | The `thakerpragnesh` repo exists but is empty |

---

## 3. Repository inventory

Verified by cloning on 10 August 2026.

| Repository | Commits | Last push | .py | .ipynb | LOC | State |
|---|---:|---|---:|---:|---:|---|
| `DeepLearning` | 60 | 2022-05-04 | 37 | 71 | 9,403 | Research scratchpad |
| `pruning_framwork_v4` | 1 | 2023-10-28 | 13 | 0 | 2,794 | Latest framework |
| `pruning_framwork_v3` | 1 | 2023-07-02 | 13 | 6 | 2,766 | Superseded |
| `pruning_framwork_v2` | 4 | 2023-01-29 | 14 | 6 | 3,206 | Superseded |
| `pruning_framwork` | 9 | 2022-12-02 | 13 | 6 | 2,952 | Superseded |
| `pruning` | 2 | 2022-04-26 | 7 | 3 | 1,285 | Superseded |
| `channelpruning` | 0 | — | 0 | 0 | 0 | **Empty** |
| `PruningKernel` | 0 | — | 0 | 0 | 0 | **Empty** |
| `PycharmPruning` | 0 | — | 0 | 0 | 0 | **Empty** |
| `DeeplearningProjects` | 0 | — | 0 | 0 | 0 | **Empty** |
| `thakerpragnesh` | 0 | — | 0 | 0 | 0 | **Empty — this is your profile README repo** |

Not cloned in the original pass (see addendum for what they actually contain):
`Algorithms-Notes`, `CppProgram`, `Programming`, `WebProject`, `WebProject1`,
`hello-world`, `uv`, and the Nielsen fork.

**Every one of the eleven repositories above lacks a README, a `.gitignore`, and a
LICENSE.** Without a license, the code is legally "all rights reserved" — nobody
can use it, which defeats the purpose of publishing it alongside papers.

"Framwork" is misspelled in four repository names.

---

## 4. Code audit: the Max-3 provenance problem

This is the most consequential finding, and it changed as the investigation went on.

### What each version actually contains

```
pruning_framwork      2022-12-02   Max-3: NO  — compute_saliency_score_channel is pure L1
pruning_framwork_v2   2023-01-29   Max-3: NO  — identical L1 implementation
pruning_framwork_v3   2023-07-02   Max-3: NO  — identical L1 implementation
pruning_framwork_v4   2023-10-28   Max-3: YES — but incorrectly wired
```

In v1–v3 the function is honest and correct: both `channel_norm` and
`channel_norm_temp` hold the L1 norm, selection reads `channel_norm_temp`, and the
result is straightforward L1-magnitude pruning. No bug — but also no Max-3.

### The v4 wiring bug

v4 adds the Max-3 loop, writing the result into `channel_norm`. Selection was never
updated to match:

```python
channel_norm      = torch.norm(tensor_t, p=1, dim=dim_to_prune)   # L1
channel_norm_temp = torch.norm(tensor_t, p=1, dim=dim_to_prune)   # L1, never overwritten
...
channel_norm[i] += max1+max2+max3                    # Max-3 written HERE
...
if channel_norm_temp[min_idx] > channel_norm_temp[j]: # selection reads TEMP (still L1)
    min_idx = j
score_value.append([min_idx, channel_norm[min_idx]])  # Max-3 only *reported*
```

Verified empirically on an 8×4×3×3 tensor:

```
channels picked to prune : [4, 3, 6]
lowest-3 by plain L1 norm: [4, 3, 6]     ← identical
```

The Max-3 score is computed, stored, printed, and ignored. **As written, v4 performs
L1-magnitude pruning.**

### What this means for the papers — the good news

The published results are almost certainly fine. In *Channel Pruning of Transfer
Learning Models Using Novel Techniques* (IEEE Access 2024), Max3 and L1 produce
different accuracy trajectories:

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Max3 acc. drop | 0.02 | 0.04 | 0.08 | 0.16 | 0.26 | 0.39 | 0.51 | 0.65 | 0.80 | 0.97 | 1.12 |
| L1 reg. acc. drop | 0.05 | 0.12 | 0.21 | 0.28 | 0.40 | 0.54 | 0.69 | 0.80 | 0.91 | 1.05 | — |

If the paper had used v4's code, these curves would be identical. They are not, and
Max3 survives one iteration longer. So working Max-3 code existed.

**It just isn't on GitHub.** Paper 1 was received May 2024, seven months after v4 was
pushed. The real implementation lives on a local machine, in Colab, or somewhere
else unpublished. Confirmed independently in the addendum: it is not in `DeepLearning`
either — every notebook and script there predates v4 and contains no Max-k code.

### Why this matters

Your most-cited methodological contribution currently has:
- no correct public implementation, and
- one public implementation that silently does something else

Anyone who tries to reproduce Max-3 from your GitHub will reproduce L1 and conclude
the method does nothing. `structured-pruning` (section 7) is now the reference
implementation.

---

## 5. Code audit: confirmed defects

All in `pruning_framwork_v4/facilitate_pruning.py`, all reproduced by execution,
and all re-confirmed verbatim against the live file in the addendum.

### D1 — Max-3 score never drives selection
*Severity: critical.* See section 4.

### D2 — The top-3 tracker does not track the top 3

```python
if v > max3:
    max3 = v
    if max3 > max1:                    # only rotates on a new maximum
        t = max3; max3 = max2; max2 = max1; max1 = t
```

The invariant `max1 ≥ max2 ≥ max3` breaks whenever a value lands between `max3` and
`max1`. Verified:

```
input          [10, 5, 1, 7, 6]
buggy tracker  (10.0, 0, 7.0)   sum = 17.0
true top-3     (10.0, 7.0, 6.0) sum = 23.0
```

Note `max2` collapsed to zero, and the value 6 was discarded entirely.

### D3 — Running maxima leak across input channels

`max1/max2/max3` are reset once per **output** channel, but
`channel_norm[i] += max1+max2+max3` executes inside the **input**-channel loop. The
sum is therefore accumulated repeatedly from monotonically growing values. This is
not "top-3 per kernel, summed over input channels" — it is a different quantity.

Consequence: the corrected implementation in `prunelib` is **not numerically
equivalent** to v4, by design.

### D4 — Non-square kernels lose columns

```python
for kw in range(size[2]):   # should be size[3]
```

On a 3×5 kernel, two of five columns are silently skipped. Harmless for 3×3, wrong
for anything else.

### D5 — `compute_distance_score_channel` is non-functional

```python
scale_tensor = torch.zeros_like(tensor_t)
for i in range(size[0]):
    scale_tensor = tensor_t[i] / torch.norm(tensor_t[[i]])   # rebinds, does not index
```

After the loop `scale_tensor` holds a single channel with shape `[In,k,k]`, so
`scale_tensor[i1]` indexes an input-channel slice. Raises
`IndexError: index 3 is out of bounds`.

Also in the same function: `for prune_amount in range(...)` shadows the function
parameter of the same name.

### D6 — `deep_model_copy_channelwise` copies nothing

```python
if torch.norm(source_model.features[l]._parameters['weight'][out_ch_old] != 0):
```

The `!= 0` sits **inside** `torch.norm`, so this computes the norm of a boolean
tensor — almost always non-zero, so the condition is effectively always true.
Separately, `out_ch_new` is never incremented, so every surviving channel is written
to destination index 0. Confirmed present with identical logic in both
`pruning_framwork_v4/facilitate_pruning.py` and `channel_pruning_saliency.py`.

### D7 — Off-by-one in `deep_copy_kernelwise`

`fin_new = fin_org` followed by `fin_new += 1` before first use.

### Summary

| ID | Location | Severity | Effect |
|---|---|---|---|
| D1 | `compute_saliency_score_channel` | Critical | Method silently reduces to L1 |
| D2 | same | High | Score is not the top-3 sum |
| D3 | same | High | Score is a different quantity entirely |
| D4 | same | Low | Non-square kernels mis-scored |
| D5 | `compute_distance_score_channel` | Critical | Raises `IndexError` |
| D6 | `deep_model_copy_channelwise` | Critical | Model copy produces garbage |
| D7 | `deep_copy_kernelwise` | Medium | Off-by-one |

---

## 6. Hygiene and privacy

### Privacy — act on this first

`DeepLearning/Major Project/` contains four directories named after individual
students:

```
G1 Gagandeep Bhgyashri/
G2 Ritik Naman/
G3 Tanmay Yash/
G4 Sagar Ashis/          (contains Pruning_RNN — likely the ADCIS 2022 work)
```

Named third parties with their coursework in a public repository is not yours to
publish. Note that `G4 Sagar Ashis` corresponds to a co-author on the ADCIS paper,
which makes the attribution question sharper rather than softer. **Addendum finding:**
this folder also contains `Student feeedback.docx` and `Student feeedback.odt` —
actual feedback documents, not just named folders. Worse than originally scoped.

### Build artefacts committed

```
DeepLearning/.ipynb_checkpoints          pruning_framwork/.idea
DeepLearning/Compression/.ipynb_checkpoints    pruning_framwork_v2/.idea
DeepLearning/my_utils/__pycache__        pruning_framwork_v2/__pycache__
pruning/my_utils/__pycache__             pruning_framwork_v3/.idea
...                                      pruning_framwork_v3/__pycache__
                                         pruning_framwork_v4/__pycache__
```

Present in six of eleven repositories. None has a `.gitignore`.

### Hardcoded absolute paths

`config.ini` and several modules embed machine-specific paths:

```
/home/pragnesh/Dataset/
/home3/pragnesh/Model/
/home/pragnesh/Logs/result.log
/home/pragnesh/Dataset/Intel_Image_Classifacation_v2/...
```

Nothing runs on another machine without editing source. (Also note
"Classifacation" is misspelled in the path.)

### Naming

- `framwork` misspelled in four repository names
- `00_MakeDasetV2.ipynb` — "Daset"
- `12 VGGNetPruning.ipynb`, `13 …`, `14 …`, `15 …` — version control by filename
- `16.ipynb` — no descriptive name
- a top-level folder named `other`
- inconsistent separators: `04_VGG_Net16_BasicPruning` vs `07 IterativeChannelPruning`

---

## 7. The replacement: structured-pruning

A clean package was built to supersede the four framework repos. This is the
repository you're reading this file from.

```
structured-pruning/
├── README.md, KT.md, PUBLICATIONS.md     docs — start with KT.md if you're new
├── CITATION.cff, LICENSE (MIT)
├── pyproject.toml, .github/workflows/tests.yml  (CI: Python 3.9-3.12)
├── prunelib/
│   ├── saliency.py               Max-k (correct), L1, L2, random
│   ├── surgery.py                conv/BN/FFN structural surgery
│   ├── scanners.py               weight distance + co-activation
│   └── evaluate.py               params, encoder-params, measured latency
├── experiments/
│   ├── 00_demo.py                runs in seconds
│   ├── 01_vgg_cifar10_sweep.py   Max3 vs L1 vs L2 vs random; --smoke / --tiny-check / full
│   ├── 02_bert_sst2_sweep.py     FFN pruning, --smoke against real transformers classes
│   ├── 03_head_redundancy.py     head similarity across layers
│   ├── 04_coactivation.py        activation-based redundancy demo
│   └── 05_ordering.py            does the CNN ordering result transfer?
├── results/
└── tests/                        16 tests, each naming the defect it guards
```

Every defect in section 5 is fixed and pinned by a test. See `KT.md` section 6 for
an honest list of what's still a gap (K-Means clustering selection is not yet
implemented; full CIFAR-10/BERT runs haven't been executed against real data yet).

Measured output (fresh run, 11 August 2026 — re-run `experiments/00_demo.py`
yourself before citing a number; latency is hardware-dependent):

```
CNN block, prune 50%:   params 38,848 → 19,456 (49.9%)   latency 5.52ms → 2.59ms (2.13x)
```

Three design decisions worth knowing (see `KT.md` section 3 for the full table
mapping every decision to the defect it prevents):

- **`prune_conv_bn` carries BatchNorm running statistics across.** The original
  omitted this; the forward pass still ran, so the resulting accuracy loss was
  silent.
- **`prune_ffn_block` validates the seam** and raises on mismatch rather than
  producing correctly-shaped nonsense.
- **`CoActivationScanner.jaccard()` filters by firing rate.** Units firing on >90%
  of tokens overlap ~96% by construction; excluding them prevents a class of false
  redundancy findings.

Every experiment has a `--smoke` flag that exercises the full pipeline on synthetic
data with no downloads, so the machinery is verifiable in seconds before committing
GPU hours. `experiments/01` additionally has `--tiny-check`, which runs the real
`torchvision.models.vgg16` class against `FakeData` — no downloads, but a much
closer proxy for the real run than `--smoke`.

---

## 8. Action plan

### Today — under one hour, highest return

1. **Remove the student directories and feedback documents.**
   ```bash
   cd DeepLearning
   git rm -r --cached "Major Project"
   echo "Major Project/" >> .gitignore
   git commit -m "Remove student project directories"
   git push
   ```
2. **Fix the bio** →
   *"AI Engineer at Simprosys. Ph.D. (NITK Surathkal) in neural network compression
   — structured pruning, quantization, and LLM evaluation."*
3. **Unpin the Nielsen fork.** Pin nothing rather than someone else's book.
4. **Fix location and website fields.**
5. **Push the profile README** to the empty `thakerpragnesh` repo.

### This week

6. **Push `structured-pruning`**, then set description, topics
   (`pytorch` `model-compression` `pruning` `transformers` `efficient-inference`),
   and pin it.
7. **Verify CI is green.**

### This month

8. **Run the real experiments** and fill the README results table.
   ```bash
   python experiments/01_vgg_cifar10_sweep.py          # needs internet access
   python experiments/02_bert_sst2_sweep.py             # needs internet access, not yet wired past --smoke
   ```
   Report what you find, including if Max-3 loses to L1 or random.
9. **Archive the four `framwork` repos** with `ARCHIVE_NOTICE.md`, then
   Settings → Danger Zone → Archive. Archive or delete the five empty ones, plus
   (per the addendum) `uv` (a fork, not your work) and the 2016–2018 coursework
   repos if you want a fully curated profile.
10. **Add a one-line README to `DeepLearning`** describing it as a research
    scratchpad and pointing to the new repo, then archive it too.

### Do not

- **Do not delete** the old repos. Archiving preserves history and reads as
  consolidation; deletion looks like something was hidden.
- **Do not try to clean `DeepLearning` in place.** Untangling 60 commits of
  scratchpad costs more than starting fresh and still leaves a messy history.

---

## 9. Open questions

**Where is the working Max-3 code?** Not in any public repository — reconfirmed
independently in the addendum by checking every notebook in `DeepLearning`, not
just the framework repos. Check local machines, Colab drive, and any NITK lab
storage. If it cannot be found, `structured-pruning`'s `max_k_saliency` is the
reference implementation going forward.

**Which code produced the ICCCNT 2023 hybrid-sequencing results?** v3 (July 2023)
is the closest by date but contains no Max-3 and no working distance function.
Same question, same resolution.

**Does the ADCIS 2022 RNN work live in `G4 Sagar Ashis/Pruning_RNN`?** If so, it
should move to its own properly-attributed repository with the co-authors credited,
not sit inside a student coursework folder.

---

## 10. Addendum — second independent verification pass

Run on 11 August 2026 by cloning all 19 public repositories fresh (rather than
relying on this document), specifically to check whether anything above was
mis-transcribed and whether anything new turns up.

**Everything above checked out exactly.** D1–D7 were re-confirmed verbatim
against the live `facilitate_pruning.py` (including the exact line-level bugs).
`DeepLearning`'s own `my_utils/facilitate_pruning.py` and all 16 of its notebooks
were checked directly for any Max-k code — none exists anywhere, and everything
there predates even `pruning_framwork_v4` by over a year, further confirming the
"unpublished, not lost in a different repo" conclusion.

**New findings not in the original pass:**

- `DeepLearning/Major Project/` contains `Student feeedback.docx` and
  `Student feeedback.odt` in addition to the four student-named folders —
  folded into section 6 above.
- `uv`, previously uncloned, is not a similarly-named project — it's an actual
  synced fork of `astral-sh/uv` (9,538 commits, hundreds of authors, none of them
  Pragnesh). A second fork bloating the profile alongside the Nielsen one.
- `CppProgram`, `Programming`, `WebProject`, `hello-world` are 2016–2018
  coursework/tutorial artifacts (basic C programs, the GitHub "hello world"
  starter). Not harmful, but noise around the one repository that represents
  current work — candidates for archiving alongside the `pruning_framwork*` repos.
- `Algorithms-Notes`, `WebProject1`, and the previously-known five empty repos
  are confirmed genuinely empty on direct clone (git warns "cloned an empty
  repository" for each).

No findings from the original pass were contradicted or need correction.
