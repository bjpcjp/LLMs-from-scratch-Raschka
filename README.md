# Building Large Language Models from Scratch (Raschka, 2024)

## Chap 1: Understanding LLMs
- Basics
- Applications
- Building & Using
- Transformer architecture
- Example datasets
- GPT architecture
- LLM buildplan

## Chap 2: Text data
- Word embeddings
- Text tokens
- Tokens --> token IDs
- Special context tokens
- Byte pair encoding
- Sliding-window data sampling
- Token embeddings
- Word position encoding

## Chap 3: Attention
- The long-sequence problem
- Capturing data dependencies with attention
- Self-attention
    - with trainable weights
    - weights for _all_ input tokens
- Self-attention with trainable weights
    - Computation
    - Python class definition
- Causal attention
    - masking
    - dropout
    - Python class definition
- Multihead attention
    - stacking single-head attention layers
    - weight splits

## Chap 4: GPT from scratch
- Architecture code
- Layer normalization
- Feed-forward nets with GELU (Gaussian error linear unit) activations
- Shortcut connections
- Attention & linear layers in a transformer block
- Model code
- Generating text

## Chap5: Pretraining - unlabeled data
- Evaluating generative text models
    - Using GPT to generate text
    - Text generation loss
    - Training & validation set loss
- LLM training
- Reducing randomness
    - Temperature scaling
    - top-k sampling
    - Modifying the text generator
- PyTorch: model file load/save
- Loading pretrained weights from OpenAI

## Chap6: Finetuning for classification
- Instruction- vs Classification-finetuning
- Dataset prep
- Dataloaders
- Initializing with pretrained weights
- Classification head
- Classification loss & accuracy
- Finetuning - supervised data
- LLM as a spam classifier

## Chap7: Instruction finetuning
- TODO

## Appendix A: Intro to PyTorch
- What is PyTorch?
    - 3 core components
    - deep learning, defined
    - installation
- Tensors
    - scalars, vectors, matrices, tensors
    - datatypes
    - common tensor ops
- Model as computation graphs
- Auto differentiation
- Designing multilayer neural nets
- Designing data loaders
- Typical training loops
- Model load/save
- GPUs and training performance
    - PyTorch on GPUs
    - Single-GPU training
    - Multi-GPU training
        - Selecting available GPUs
- Resources
- Exercise answers

## Appendix B: References
- Chapter 1
    - [Bloomberg GPT](https://arxiv.org/abs/2303.17564)
    - [Medical Q&A with LLMs](https://arxiv.org/abs/2305.09617)
    - [Attention is all you need](https://arxiv.org/abs/1706.03762)
    - [BERT (original encoder-style transformer)](https://arxiv.org/abs/1810.04805)
    - [decoder-style GPT3 model](https://arxiv.org/abs/2005.14165)
    - [original vision transformer](https://arxiv.org/abs/2010.11929)
    - [RWKV](https://arxiv.org/abs/2305.13048)
    - [Hyena hierarchy](https://arxiv.org/abs/2302.10866)
    - [Mamba](https://arxiv.org/abs/2312.00752)
    - [Llama2](https://arxiv.org/abs/2307.09288)
    - ["The Pile" text dataset](https://arxiv.org/abs/2101.00027)
    - [GPT-3 finetuning](https://arxiv.org/abs/2203.02155)

- Chapter 2
    - [Machine Learning Q and AI (2023) by Sebastian Raschka](https://leanpub.com/machine-learning-q-and-ai)
    - [byte pair encoding](https://arxiv.org/abs/1508.07909)
    - [OpenAI byte pair encoding tokenizer for GPT-2](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
    - [OpenAI byte pair encoding, illustrated](https://platform.openai.com/tokenizer)
    - [BPE tokenizer from scratch (Karpathy)](https://github.com/karpathy/minbpe)
    - [SentencePiece Tokenizer and Detokenizer](https://aclanthology.org/D18-2012/)
    - [Fast WordPiece Tokenization](https://arxiv.org/abs/2012.15524)

- Chapter 3
    - [Bahdanau attention](https://arxiv.org/abs/1409.0473)
    - [self-attention as scaled dot-product attention](https://arxiv.org/abs/1706.03762)
    - [FlashAttention](https://arxiv.org/abs/2205.14135)
    - [FlashAttention-2](https://arxiv.org/abs/2307.08691)
    - [PyTorch self-attention and causal attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
    - [PyTorch MultiHeadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
    - [Dropout](https://jmlr.org/papers/v15/srivastava14a.html)
    - [Simplifying Transformer Blocks](https://arxiv.org/abs/2311.01906)

- Chapter 4
    - [Layer Normalization](https://arxiv.org/abs/1607.06450)
    - [Pre-LayerNorm](https://arxiv.org/abs/2002.04745)
    - [ResiDual](https://arxiv.org/abs/2304.14802)
    - [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
    - [GELU (Gaussian Error Linear Unit) activation](https://arxiv.org/abs/1606.08415)
    - [GPT-2—124M, 355M, 774M, and 1.5B parameters](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
    - [GPT-3 (OpenAI)](https://arxiv.org/abs/2005.14165)
    - [GPT-3 Technical Overview (Lambda Labs)](https://lambdalabs.com/blog/demystifying-gpt-3)
    - [NanoGPT](https://github.com/karpathy/nanoGPT)
    - [Feedforward computation](https://www.harmdevries.com/post/context-length/)

- Chapter 5
    - [loss functions & log transformations](https://www.youtube.com/watch?v=GxJe0DZvydM)
    - [Pythia](https://arxiv.org/abs/2304.01373)
    - [OLMo](https://arxiv.org/abs/2402.00838)
    - [Project Gutenberg for LLM training](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/03_bonus_pretraining_on_gutenberg)
    - [Simple and Scalable Pretraining Strategies](https://arxiv.org/abs/2403.08763)
    - [BloombergGPT](https://arxiv.org/abs/2303.17564)
    - [GaLore optimizer](https://arxiv.org/abs/2403.03507)
    - [GaLore code repo](https://github.com/jiaweizzhao/GaLore)
    - [Dolma: an Open Corpus of Three Trillion Tokens](https://arxiv.org/abs/2402.00159)
    - [The Pile](https://arxiv.org/abs/2101.00027)
    - [RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116)
    - [RedPajama](https://github.com/togethercomputer/RedPajama-Data)
    - [FineWeb dataset from CommonCrawl](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
    - [top-k sampling](https://arxiv.org/abs/1805.04833)
    - [Beam search (not cover in chapter 5)](https://arxiv.org/abs/1610.02424)

- Chapter 6
    - [Finetuning Transformers](https://magazine.sebastianraschka.com/p/using-and-finetuning-pretrained-transformers)
    - [Finetuning LLMs](https://magazine.sebastianraschka.com/p/finetuning-large-language-models)
    - [More spam classification experiments](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch06/02_bonus_additional-experiments)
    - [Binary classification using a single output node](https://sebastianraschka.com/blog/2022/losses-learned-part1.html)
    - [Imbalanced-learn user guide](https://imbalanced-learn.org/stable/user_guide.html)
    - [spam email classification dataset](https://huggingface.co/datasets/TrainingDataPro/email-spam-classification)
    - [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
    - [RoBERTa](https://arxiv.org/abs/1907.11692)
    - [IMDB Movie Reviews sentiment](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/03_bonus_imdb-classification)
    - [causal mask removal](https://arxiv.org/abs/2310.01208)
    - [LLM2Vec](https://arxiv.org/abs/2404.05961)

## Appendix C: Exercise Solutions
(see jupyter notebook)

## Appendix D: Training loop bells & whistles
- Learning rate warmup
- Cosine decay
- Gradient clipping
- Modified training function

## Appendix E: parameter-efficient finetuning with LoRA
- Intro
- Dataset prep
- Model init
- LoRA
- 