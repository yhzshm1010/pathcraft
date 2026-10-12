# Dataset sources and third-party notices

These are derived research TRAIN supervision snapshots, not newly collected
benchmark questions. Original dataset content retains its upstream terms;
this repository does not grant a blanket MIT or unrestricted commercial
license over CWQ/WebQSP questions and annotations. Consult the original
distributions for applicable terms and obtain any additional permissions
required for your intended use.

## ComplexWebQuestions (CWQ)

Alon Talmor and Jonathan Berant. 2018. *The Web as a Knowledge-base for
Answering Complex Questions.* NAACL.

- [Original paper](https://aclanthology.org/N18-1059/)
- [Original project](https://github.com/alontalmor/WebAsKB)
- [Dataset website](https://www.tau-nlp.sites.tau.ac.il/compwebq)

The published 25,328-example stage bundle preserves the retained TRAIN
snapshot and appends strict Gold-SPARQL-derived TRAIN supervision. Historical
schema normalization is already part of this retained snapshot; publication
does not make new changes to it.

## WebQuestions Semantic Parses (WebQSP)

Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang and Jina Suh.
2016. *The Value of Semantic Parse Labeling for Knowledge Base Question
Answering.* ACL.

- [Original paper](https://aclanthology.org/P16-2033/)
- [Microsoft dataset distribution and accompanying terms](https://www.microsoft.com/en-us/download/details.aspx?id=52763)

The published 2,651-example stage bundle is the final four-stage TRAIN export,
including separate Operator supervision. Microsoft/upstream dataset terms
remain applicable; KaeDe's software license is not a replacement for them.

## KaeDe-derived supervision

The data-construction workflow uses KaeDe's released TRAIN decompositions,
logical forms and processed annotations, where applicable.

- [KaeDe repository](https://github.com/pvfeldt/KaeDe)
- [KaeDe paper](https://aclanthology.org/2025.findings-emnlp.582/)
- [Preserved upstream MIT notice](licenses/KaeDe-MIT.txt)

The MIT notice applies to the relevant KaeDe contributions. Original benchmark
questions and other upstream content are not relicensed by this notice.
