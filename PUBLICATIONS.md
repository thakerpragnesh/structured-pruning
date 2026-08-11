# Publications

Six peer-reviewed publications from my Ph.D. work at NITK Surathkal
(2018–2025), supervised by Dr. Biju R. Mohan. This package implements or
extends the methods from papers 1–3 directly.

## 1. Channel Pruning of Transfer Learning Models Using Novel Techniques

Thaker, P. and Mohan, B. R. *IEEE Access*, vol. 12, pp. 94914–94925, 2024.
DOI: [10.1109/ACCESS.2024.3416997](https://doi.org/10.1109/ACCESS.2024.3416997)

Introduces the Max-3 saliency criterion (`prunelib.saliency.max_k_saliency`),
compared against K-Means clustering and SVD on VGG16 and ResNet56 (CIFAR-10).
Max-3: 46.19% param / 61.91% FLOPs reduction on VGG16, 35.15% on ResNet56,
both under a 1% accuracy-drop threshold — ahead of K-Means (40.00%/49.20%)
and SVD (20.07%/24.64%) on VGG16.

```bibtex
@article{thaker2024channel,
  author  = {Thaker, Pragnesh and Mohan, Biju R.},
  title   = {Channel Pruning of Transfer Learning Models Using Novel Techniques},
  journal = {IEEE Access},
  volume  = {12},
  pages   = {94914--94925},
  year    = {2024},
  doi     = {10.1109/ACCESS.2024.3416997}
}
```

## 2. Enhancing Deep Compression of CNNs: A Novel Regularization Loss and the Impact of Distance Metrics

Thaker, P. and Mohan, B. R. *IEEE Access*, vol. 12, pp. 172537–172547, 2024.
DOI: [10.1109/ACCESS.2024.3498901](https://doi.org/10.1109/ACCESS.2024.3498901)

Introduces the CSD (Custom Standard Deviation) group regularization loss —
ratio of L1 norm to CSD per channel — for structured pruning: 46.14% param /
61.91% FLOPs reduction on VGG16. Also evaluates Manhattan vs. Euclidean vs.
Cosine distance for K-Means-based channel selection; Manhattan wins
(35.15%/49.11% vs. 31.01%/43.96% Euclidean, 21.93%/32.03% Cosine).
`prunelib.scanners.pairwise_distance_matrix` implements all three metrics.

```bibtex
@article{thaker2024enhancing,
  author  = {Thaker, Pragnesh and Mohan, Biju R.},
  title   = {Enhancing Deep Compression of CNNs: A Novel Regularization Loss and the Impact of Distance Metrics},
  journal = {IEEE Access},
  volume  = {12},
  pages   = {172537--172547},
  year    = {2024},
  doi     = {10.1109/ACCESS.2024.3498901}
}
```

## 3. Comparing Different Sequences of Pruning Algorithms for Hybrid Pruning

Thaker, P. and Mohan, B. R. *14th IEEE International Conference on Computing,
Communication and Networking Technologies (ICCCNT)*, IIT Delhi, 2023.
DOI: [10.1109/ICCCNT56998.2023.10307846](https://doi.org/10.1109/ICCCNT56998.2023.10307846)

Shows pruning order matters: channel-saliency → channel-similarity →
kernel-saliency reaches 58.44% parameter reduction (42.83% FLOPs) on VGG16 /
Intel Image Classification, while any kernel-first sequence caps at 28.14%,
because kernel sparsity perturbs the channel statistics channel selection
depends on. `experiments/05_ordering.py` demonstrates the underlying scoring
mechanism.

```bibtex
@inproceedings{thaker2023comparing,
  author    = {Thaker, Pragnesh and Mohan, Biju R.},
  title     = {Comparing Different Sequences of Pruning Algorithms for Hybrid Pruning},
  booktitle = {2023 14th International Conference on Computing Communication and Networking Technologies (ICCCNT)},
  year      = {2023},
  doi       = {10.1109/ICCCNT56998.2023.10307846}
}
```

## 4. Compression of Convolution Neural Network Using Structured Pruning

Thaker, P. and Mohan, B. R. *IEEE 7th International Conference for
Convergence in Technology (I2CT)*, Pune, 2022.
DOI: [10.1109/I2CT54291.2022.9825302](https://doi.org/10.1109/I2CT54291.2022.9825302)

```bibtex
@inproceedings{thaker2022compression,
  author    = {Thaker, Pragnesh and Mohan, Biju R.},
  title     = {Compression of Convolution Neural Network Using Structured Pruning},
  booktitle = {2022 IEEE 7th International Conference for Convergence in Technology (I2CT)},
  year      = {2022},
  doi       = {10.1109/I2CT54291.2022.9825302}
}
```

## 5. Kernel-Level Pruning for CNN

Thaker, P. and Mohan, B. R. In *Machine Intelligence Techniques for Data
Analysis and Signal Processing*, Lecture Notes in Electrical Engineering
vol. 997, pp. 71–78, Springer Nature Singapore, 2023.
DOI: [10.1007/978-981-99-0085-5_6](https://doi.org/10.1007/978-981-99-0085-5_6)

```bibtex
@incollection{thaker2023kernel,
  author    = {Thaker, Pragnesh and Mohan, Biju R.},
  title     = {Kernel-Level Pruning for CNN},
  booktitle = {Machine Intelligence Techniques for Data Analysis and Signal Processing},
  series    = {Lecture Notes in Electrical Engineering},
  volume    = {997},
  pages     = {71--78},
  publisher = {Springer Nature Singapore},
  year      = {2023},
  doi       = {10.1007/978-981-99-0085-5_6}
}
```

## 6. Comparative Study of Pruning Techniques in Recurrent Neural Networks

Choudhury, S., Rout, A. K., Thaker, P. and Mohan, B. R. *International
Conference on Advances in Data-driven Computing and Intelligent Systems
(ADCIS)*, 2022.

```bibtex
@inproceedings{choudhury2022comparative,
  author    = {Choudhury, Sagar and Rout, Ashish Kumar and Thaker, Pragnesh and Mohan, Biju R.},
  title     = {Comparative Study of Pruning Techniques in Recurrent Neural Networks},
  booktitle = {International Conference on Advances in Data-driven Computing and Intelligent Systems (ADCIS)},
  year      = {2022}
}
```

**Note on provenance:** if this RNN work is the code currently sitting in
`DeepLearning/Major Project/G4 Sagar Ashis/Pruning_RNN` (Sagar Choudhury is a
co-author here), it deserves its own properly-attributed repository rather
than living inside a student-coursework folder. See `GITHUB_AUDIT.md`,
section 9.
