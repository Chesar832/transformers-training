# BERT and its variants
![banner](https://hoangluyen.com/b/3459/p/1679307317.jpg)

This `.md` file contains 26 questions about BERT and its variants which will be used to get a deep understanding of this topic.

---

#### Understanding BERT

What does BERT stand for, and what is its primary purpose in natural language processing (NLP)?

**BERT** stands for **Bidirectional Encoder Representations from Transformers**. It is a pre-trained deep learning model designed primarily for **natural language understanding** tasks in NLP.

Its **primary purpose** is to:
1. **Understand the context** of a word in a sentence by considering both the words that come **before and after** it, making it **bidirectional**.
2. It excels in tasks like **text classification**, **sentiment analysis**, **question answering**, and other tasks that require a deep understanding of the context of text.


---

#### Pre-training Tasks

What are the two main pre-training objectives used in BERT, and how do they contribute to its language understanding capabilities?

BERT uses two main **pre-training objectives** to enhance its language understanding capabilities:

1. **Masked Language Model (MLM)**
   - **Description**: In this task, BERT randomly masks some of the words in a sentence and tries to predict the **masked words** based on the context provided by the other words.
   - **Example**:
     - Suppose we have the sentence: 
       > "The quick brown fox jumps over the lazy dog."
     - During pre-training, BERT might mask two words:
       > "The quick [MASK] fox [MASK] over the lazy dog."
     - BERT will then try to predict that the first `[MASK]` should be **"brown"** and the second `[MASK]` should be **"jumps"** based on the context of the surrounding words.
   - **Contribution**: This approach encourages BERT to learn **contextual relationships** by considering words **both before and after** the masked words, improving its ability to understand the context within a sentence.

2. **Next Sentence Prediction (NSP)**
   - **Description**: In this task, BERT receives pairs of sentences and learns to predict whether the second sentence is likely to follow the first in a coherent context.
   - **Example**:
     - **Sentence A**: "The weather was nice yesterday."
     - **Sentence B** (positive pair): "We decided to have a picnic in the park."
     - **Sentence B** (negative pair): "I am learning about quantum physics."
     - BERT will train to predict that the second sentence in the **positive pair** logically follows the first sentence, while the second sentence in the **negative pair** does not.
   - **Contribution**: This helps BERT capture **sentence-level relationships**, making it more effective in understanding the flow of information across sentences, which is crucial for tasks like **question answering** and **sentence-pair classification**.

Together, these objectives enable BERT to develop a deeper **semantic understanding** of text, making it well-suited for a variety of **downstream NLP tasks**.

---

#### Bidirectional Encoding

How does BERT's bidirectional approach differ from traditional unidirectional language models, and why is this significant?

**Traditional Unidirectional Language Models:**
- **Description**: Unidirectional language models process text in **one direction**—either left-to-right (e.g., GPT) or right-to-left (e.g., some early language models).
- **Example**:
  - In a left-to-right model, given the sentence:
    > "The quick brown fox jumps over the lazy dog."
  - The model would predict each word by considering **only the words to its left**. So, when predicting "fox," it can only use the context of "The quick brown" and not the words after "fox."
- **Limitation**: This directional constraint limits the model’s ability to fully understand the **entire context** of the sentence, which can be crucial for tasks like named entity recognition or sentiment analysis where the full sentence context is needed for accurate predictions.

**BERT’s Bidirectional Approach:**
- **Description**: BERT processes text in a **bidirectional** manner, considering **both left and right contexts** simultaneously. It can analyze all words in a sentence at once, allowing for a deeper understanding of each word’s meaning.
- **Example**:
  - Using the same sentence:
    > "The quick brown fox jumps over the lazy dog."
  - When predicting "fox," BERT will use both the **preceding words ("The quick brown")** and the **following words ("jumps over")** to understand its context more accurately.
- **Significance**:
  1. **Enhanced Contextual Understanding**: By processing words in both directions, BERT captures the **full context** of each word in a sentence, leading to better performance in tasks requiring complex comprehension, such as **question answering**, **sentence classification**, and **text entailment**.
  2. **Better Handling of Ambiguity**: Bidirectional encoding helps BERT resolve ambiguous words or phrases by considering the surrounding context from all sides, which traditional unidirectional models struggle with.
  3. **Improved Accuracy in Downstream Tasks**: As a result of its comprehensive context understanding, BERT achieves **state-of-the-art performance** in a variety of natural language processing tasks, outperforming many unidirectional models.

In summary, BERT's bidirectional approach enables a **deeper and more nuanced understanding** of language, making it significantly more effective in capturing context compared to traditional unidirectional models.


---

#### Tokenization with WordPiece

What is WordPiece tokenization, and why is it used in BERT?


**WordPiece tokenization** is a subword-based tokenization technique used in BERT to break down words into smaller units (subwords) to better handle vocabulary, rare words, and unknown words.

**What is WordPiece Tokenization?**
- WordPiece tokenization splits words into subword units based on frequency in the training corpus. It breaks down words into smaller, more common subwords, which are later combined to form the original word.
- **Example**:
  - Consider the word "unbelievable":
    - WordPiece might split it into: 
      > ["un", "##believ", "##able"]
    - Here, "un" is treated as a prefix, while "##believ" and "##able" are subword tokens.
  - The "##" prefix indicates that the token is part of a word and not a standalone word itself.

**Why is WordPiece Tokenization Used in BERT?**
1. **Efficient Handling of Rare and Unknown Words**:
   - **Problem with traditional tokenization**: In many tokenization schemes, rare words or misspellings result in the token being classified as "unknown," losing valuable context.
   - **Solution with WordPiece**: WordPiece decomposes rare words into meaningful subwords, allowing BERT to understand their components and retain context. For example:
     - The rare word "bioluminescence" might be tokenized as: 
       > ["bio", "##lumines", "##cence"]
     - Even if the full word is uncommon, its components ("bio", "##lumines", "##cence") provide useful semantic information.

2. **Reduced Vocabulary Size**:
   - WordPiece tokenization reduces the overall vocabulary size by using common subwords rather than whole words.
   - **Benefit**: This makes the model more efficient and faster to train, while still capturing the semantic meaning of words.

3. **Better Generalization**:
   - By breaking words into subwords, WordPiece tokenization allows BERT to generalize better across different word forms.
   - **Example**: The words "running," "runner," and "ran" share the common root "run," which WordPiece can identify and use to understand the semantic relationship between these forms.

4. **Improved Context Understanding**:
   - Since WordPiece tokenization breaks down words into meaningful components, it allows BERT to understand morphological patterns, such as prefixes, suffixes, and stems, leading to more accurate language understanding.

**In summary**, WordPiece tokenization helps BERT handle **rare words, reduce vocabulary size**, and **improve generalization** by breaking down words into subword units, making it more versatile and effective for various NLP tasks.

---

#### Fine-tuning BERT for Downstream Tasks

What are the general steps involved in fine-tuning BERT for a specific NLP task, such as text classification or named entity recognition?

The full process is in this ![notebook](./fine-tune-bert-for-sentiment-classification.ipynb) and the used data was this ![one](./reviews.csv).

---

#### Handling Out-of-Vocabulary (OOV) Words

How does BERT handle words that are not present in its vocabulary during tokenization?

BERT uses **WordPiece tokenization** to handle words that are not present in its vocabulary during tokenization. This approach ensures that even unknown words can be represented meaningfully.

**How BERT Handles OOV Words:**

1. **Subword Tokenization**:
   - If a word is not found in BERT's vocabulary, it is split into smaller subword units based on common prefixes, suffixes, and roots.
   - For example, the word "unbelievably" might not be in BERT’s vocabulary, but it can be broken down into subwords like:
     > ["un", "##believ", "##ably"]
   - The special "##" prefix indicates that the subword is part of a larger word, not a standalone token.

2. **Mitigating OOV Problems**:
   - By breaking down words into smaller, known components, BERT can **maintain semantic information** even for words it hasn’t seen during pre-training.
   - This allows BERT to effectively generalize and process words that are rare, misspelled, or newly coined.

3. **Robustness to New Words**:
   - WordPiece tokenization ensures that even if a completely new word is encountered, BERT can still understand parts of the word based on its subwords.
   - For instance, a new term like "biohacking" can be tokenized as:
     > ["bio", "##hack", "##ing"]
   - This way, BERT can leverage its knowledge of "bio" and "##hack" to make an informed prediction about the word's meaning.

**In summary**, BERT’s use of WordPiece tokenization allows it to handle OOV words by decomposing them into known subwords, ensuring better context retention and reducing information loss during tokenization.

---

#### Transformer Architecture Basics

What are the key components of the transformer architecture utilized by BERT?

The transformer architecture used by BERT is built around several core components that enable its effective understanding of language. Here are the key components:

1. **Self-Attention Mechanism**:
   - The self-attention mechanism allows BERT to focus on different parts of the input sequence simultaneously.
   - It computes a weighted sum of values based on attention scores derived from **queries**, **keys**, and **values**. These scores determine how much attention should be given to each token in the sequence relative to others.
   - This mechanism enables BERT to capture both **local** and **global** dependencies in the text, making it bidirectional.

2. **Multi-Head Attention**:
   - BERT uses **multi-head attention** to capture various aspects of the relationships between words.
   - Instead of relying on a single attention head, it uses multiple heads (e.g., 12 in BERT-Base, 16 in BERT-Large), each attending to different semantic aspects of the input sequence.
   - The outputs from all heads are concatenated and linearly transformed to provide a richer representation of the context.

3. **Positional Encoding**:
   - Since transformers lack inherent recurrence or sequence information, BERT uses **positional encodings** to inject positional information into the input embeddings.
   - Positional encodings are added to the token embeddings to help the model understand the order of words in the sequence.

4. **Layer Normalization**:
   - BERT applies **layer normalization** after each sub-layer (attention or feed-forward layer) to stabilize training and improve convergence.
   - This helps the model maintain the scale of the activations, ensuring that gradient flow remains consistent throughout the network.

5. **Feed-Forward Neural Network (FFN)**:
   - After the multi-head attention mechanism, each token representation passes through a **feed-forward neural network**.
   - This consists of two fully connected layers with a non-linear activation function (e.g., GELU) in between. It helps in transforming the attention outputs to more meaningful representations.

6. **Residual Connections**:
   - **Residual connections** are added around each sub-layer (attention or FFN) to allow gradients to flow more easily through the network, addressing potential issues with vanishing gradients.
   - They improve training stability and help maintain context across layers by adding the original input back to the transformed output.

7. **Stacked Encoder Layers**:
   - BERT uses **stacked encoders** (12 in BERT-Base, 24 in BERT-Large), where each encoder contains the components mentioned above: multi-head attention, feed-forward layers, layer normalization, and residual connections.
   - These stacked layers allow BERT to capture deeper and more complex patterns in the input text, contributing to its strong language understanding capabilities.

---

#### Positional Embeddings

Why are positional embeddings important in BERT, and how do they function within the model?

**Importance**:
- **Sequence Awareness**: Transformers process tokens independently and lack inherent awareness of word order. Positional embeddings provide this **positional context**, allowing BERT to understand the sequence of words.
- **Understanding Syntax**: Language depends on word order; without positional information, BERT cannot capture grammar, syntax, or relative positioning needed for accurate language understanding.
- Positional embeddings enable BERT to integrate **word order** into its representations, ensuring it captures both **semantic content** and **sequence structure**.

**Functionality**:
1. **Addition to Token Embeddings**:
   - Each token embedding is **added** to its corresponding positional embedding. This combination enables BERT to know both the word’s meaning and its position.
   - For instance, the embedding for "sat" at position 3 becomes:
     > `[E_sat] + [P_3]`
   
2. **Learned Embeddings**:
   - BERT uses **learned positional embeddings**, meaning it learns these positions during training, unlike sinusoidal encodings used in some other models.

---

#### Masking Strategy in MLM

In the Masked Language Modeling task, how are tokens selected for masking, and what proportion of tokens are masked during pre-training?

### Masking Strategy in Masked Language Modeling (MLM)

**Token Selection for Masking**:
- In MLM, **individual tokens** are randomly selected for masking across the sequence, meaning the selection is based on single tokens, not groups of words.
- The selection of tokens is **entirely random** without any specific criteria, ensuring that a variety of tokens are masked over different training batches.

**Proportion of Tokens Masked During Pre-Training**:
- **15% of the total tokens** in each input sequence are chosen to be masked. 
- The masking strategy is further divided as follows (in case of one token for each word, but remember that BERT uses wordpiece tokenization):
  1. **80% of the selected tokens** are replaced with the **[MASK]** token.
     - Example: "The cat sat" becomes "The [MASK] sat".
  2. **10% of the selected tokens** are replaced with a **random token** from the vocabulary.
     - Example: "The cat sat" might become "The dog sat".
  3. **10% of the selected tokens** remain **unchanged**, even though they were selected for masking.
     - Example: "The cat sat" remains "The cat sat".

**Purpose**:
- This varied approach prevents the model from overfitting to the **[MASK]** token.
- By randomly keeping some tokens unchanged or replacing them with random tokens, BERT learns to build more general and robust **contextual representations** rather than relying solely on the presence of **[MASK]**.

---

#### Next Sentence Prediction Purpose

What is the role of the Next Sentence Prediction task in BERT, and how does it help the model understand relationships between sentences?

**Role of the NSP Task**:
- The **Next Sentence Prediction (NSP)** task in BERT is designed to help the model understand the **relationships between sentences**.
- It is a **binary classification task** where BERT receives two sentences as input and predicts whether the second sentence **logically follows** the first one.
  - **Positive Pair**: The second sentence follows the first one in a coherent context.
  - **Negative Pair**: The second sentence is unrelated to the first one.

**How NSP Helps BERT Understand Sentence Relationships**:
1. **Capturing Coherence**:
   - NSP enables BERT to model **coherent connections** between consecutive sentences, making it more effective for tasks that involve multi-sentence context, such as **question answering** and **text entailment**.

2. **Sentence-Level Context**:
   - By learning to predict sentence pairs, BERT becomes more sensitive to **long-range dependencies** and sentence structure, which is crucial for tasks like **paragraph-level classification** or **document understanding**.

3. **Training Process**:
   - During training, **50%** of the input pairs are positive pairs (i.e., the second sentence follows the first), while the other **50%** are negative pairs (i.e., the second sentence is randomly chosen).
   - This balance helps BERT learn to differentiate between related and unrelated sentences, improving its performance in downstream tasks that require an understanding of sentence relationships.

---

#### Limitations of Input Length

What is the maximum input sequence length for BERT, and how does this limitation affect its application to longer texts?

**Maximum Input Sequence Length**:
- The maximum input sequence length for BERT is typically **512 tokens**. This includes special tokens like **[CLS]** and **[SEP]**, which are part of the input sequence.

**Impact on Longer Texts**:
1. **Truncation**:
   - When a text exceeds 512 tokens, it is **truncated** to fit within the limit. This can lead to the loss of important information, as parts of the text beyond the 512th token are simply discarded.
   - Truncation is particularly problematic for tasks that require understanding the full context of longer documents, such as **summarization** or **document-level classification**.

2. **Splitting Texts**:
   - To handle longer texts, the text can be split into **multiple chunks** of 512 tokens each. However, this approach can break the **contextual flow** across chunks, affecting tasks that require understanding relationships between distant parts of the text.

3. **Sliding Window Technique**:
   - For tasks requiring more context, a **sliding window approach** can be used, where overlapping windows of 512 tokens are created. This helps preserve context but increases computational costs and processing time.

4. **Performance and Efficiency**:
   - The 512-token limit means that BERT’s performance can be suboptimal on tasks that naturally require longer input sequences. Handling longer inputs requires additional strategies, which can be computationally intensive.

In summary, BERT’s 512-token limit restricts its ability to process longer texts effectively, necessitating strategies like **truncation**, **splitting**, or using a **sliding window** to manage longer sequences.

---

#### Use of Special Tokens

What are the special tokens `[CLS]` and `[SEP]` used for in BERT, and why and where are they placed in the input sequence?

**`[CLS]` Token**:
- The `[CLS]` token, short for **Classification**, is added at the **beginning** of the input sequence.
- It serves as a **global representation** of the entire sequence, making it essential for sequence-level tasks like **text classification** or **sentence-pair classification**.
- The final hidden state of the `[CLS]` token is used for the output layer during tasks where BERT needs to predict a label for the whole sequence.

**`[SEP]` Token**:
- The `[SEP]` token, which stands for **Separator**, is used to indicate the **end of a sentence** or to separate two sentences in a sentence-pair input.
- It is placed at the **end of each sentence**, allowing BERT to clearly identify sentence boundaries.
- This token is crucial for tasks like **Next Sentence Prediction (NSP)**, where BERT needs to distinguish between two distinct sentences or parts of the text.

**Placement in the Input Sequence**:

1. **Single Sentence Input**:
   - The input format is structured as:
     > `[CLS]`, "The", "cat", "sat", `[SEP]`
   - The `[CLS]` token is at the start, and the `[SEP]` token is at the end. This format is commonly used for tasks like **text classification**.

2. **Sentence-Pair Input**:
   - When BERT processes two sentences, the input is structured as:
     > `[CLS]`, "The", "cat", "sat", `[SEP]`, "It", "was", "happy", `[SEP]`
   - The `[CLS]` token marks the start of the input, and the `[SEP]` tokens are placed at the end of each sentence, separating them.
   - This structure is vital for tasks involving **sentence relationships**, such as NSP or **question answering**, where BERT needs to compare or relate two text segments.

By integrating these tokens into the input sequence, BERT effectively manages single and paired sentences, maintains context, and handles sentence-level relationships.

---

#### Transfer Learning in BERT

How does BERT leverage transfer learning, and why is this advantageous for NLP tasks with limited labeled data?

- BERT uses a two-stage **transfer learning** process: **pre-training** and **fine-tuning**.
  1. **Pre-training**: BERT is pre-trained on large, unlabeled datasets using tasks like **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)**. This step enables BERT to learn general language representations, capturing a wide range of syntactic and semantic patterns.
  2. **Fine-tuning**: Once pre-trained, BERT is further trained (fine-tuned) on a **specific, labeled dataset** for a given downstream NLP task, such as **text classification**, **named entity recognition (NER)**, or **question answering**. In this step, only a small amount of labeled data is needed to adapt BERT to the task.

**Why Transfer Learning is Advantageous for NLP Tasks with Limited Labeled Data**:

1. **Reduced Data Requirements**:
   - BERT’s pre-trained representations allow it to generalize well even with limited labeled data, as it has already learned rich language features during pre-training.
   - Fine-tuning adapts these general features to specific tasks with minimal labeled data, making BERT effective in scenarios where labeled data is scarce or expensive to obtain.

2. **Improved Performance**:
   - The pre-trained model retains broad language understanding, enabling it to perform well across a variety of tasks after fine-tuning.
   - This results in better performance compared to models trained from scratch on limited data, as BERT’s representations are already fine-tuned to capture language patterns.

3. **Efficient Training**:
   - Transfer learning with BERT reduces the computational cost and time required for training. Fine-tuning requires fewer epochs since the model starts with pre-trained weights, making it more efficient than training a model from scratch.

By leveraging transfer learning, BERT achieves **strong performance** across diverse NLP tasks, even when labeled data is limited, making it a versatile solution for real-world applications.

---

#### Differences Between BERT Base and Large

What are the main differences between BERT Base and BERT Large models in terms of architecture and performance?

BERT has two standard model variants, **BERT Base** and **BERT Large**, which differ primarily in terms of architecture and performance.

![berts](https://images.prismic.io/turing/65a53f447a5e8b1120d5891f_BERT_NLP_model_architecture_d285530efe.webp?auto=format,compress)

**1. Architectural Differences**:

| Feature           | BERT Base         | BERT Large        |
|-------------------|-------------------|-------------------|
| Layers (Encoders) | 12                | 24                |
| Hidden Size       | 768               | 1024              |
| Attention Heads   | 12                | 16                |
| Parameters        | ~110 million      | ~340 million      |

- **Layers (Encoders)**:  
  BERT Base has **12 layers**, while BERT Large has **24 layers**, making BERT Large significantly deeper, with twice the number of transformer blocks.
- **Hidden Size**:  
  The hidden size for BERT Base is **768**, while BERT Large has a hidden size of **1024**, resulting in larger embeddings and more complex representations.
- **Attention Heads**:  
  BERT Base uses **12 attention heads**, while BERT Large has **16 attention heads**, enabling it to capture more diverse aspects of the context.
- **Parameters**:  
  BERT Large has over **340 million parameters**, three times more than BERT Base, making it much more powerful but also more computationally intensive.

**2. Performance Differences**:
- **Accuracy on NLP Tasks**:  
  BERT Large typically achieves **higher accuracy** across various NLP tasks, such as **text classification**, **question answering**, and **named entity recognition**. Its deeper architecture allows for better modeling of complex patterns in the data.
- **Generalization**:  
  BERT Large generally shows better **generalization** capabilities, especially on tasks that require nuanced understanding or longer sequences, due to its larger capacity and attention mechanisms.
- **Training Time and Resources**:  
  BERT Large requires significantly more **training time**, **memory**, and **computation** compared to BERT Base, making it more challenging to fine-tune or deploy without high-performance hardware (e.g., GPUs/TPUs).

While BERT Large offers better performance and more nuanced representations, BERT Base is more resource-efficient and suitable for scenarios where computational resources are limited or faster inference is needed.

---

#### Applications of BERT

List some common NLP tasks where BERT has significantly improved performance.

The top 5 most common NLP Tasks Improved by BERT are:

1. **Text Classification**:
   - BERT is widely used for **text classification** tasks, such as **sentiment analysis**, **topic categorization**, and **spam detection**.
   - Its ability to capture context bidirectionally improves accuracy in identifying relevant features for classifying text into predefined categories.

2. **Named Entity Recognition (NER)**:
   - BERT has significantly enhanced performance in **NER** tasks, where it identifies and classifies entities like names of people, organizations, locations, and dates in text.
   - The model's contextual understanding enables it to distinguish entities more accurately, even in complex or ambiguous contexts.

3. **Question Answering (QA)**:
   - BERT excels in **question answering** systems, where it is used to extract precise answers from a given passage or document based on a query.
   - Models like **BERT-based SQuAD** are effective in extracting the most relevant spans of text, making BERT one of the top-performing models for QA.

4. **Sentence Pair Classification**:
   - BERT improves tasks like **natural language inference (NLI)**, **paraphrase detection**, and **Next Sentence Prediction (NSP)**, where relationships between two sentences need to be understood.
   - Its ability to model sentence-level relationships allows for better performance in determining whether two sentences are logically related, contradictory, or neutral.

5. **Text Summarization**:
   - While BERT is not a generative model, it is often used in **extractive summarization** to identify the most important sentences or phrases from a document.
   - BERT’s contextual embeddings help capture key information, making it effective for creating concise summaries.

These applications demonstrate BERT's versatility in handling various NLP tasks, leveraging its strong contextual understanding to deliver significant performance improvements.

---

#### Critique of Pre-training Objectives

Analyze the limitations of BERT's original pre-training objectives—Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). How do these limitations affect downstream performance, and what alternative objectives have been proposed to address them?

**1. Masked Language Modeling (MLM)**:
   - **Limitation**: MLM masks 15% of the tokens randomly, which creates a discrepancy between pre-training and fine-tuning, where no tokens are masked.
     - This leads to a **mismatch** in input processing since BERT sees `[MASK]` tokens during pre-training but not during real tasks, affecting its generalization.
   - **Contextual Gap**: MLM only provides partial context due to masking, limiting BERT's ability to learn **full-sentence coherence** during pre-training.

**2. Next Sentence Prediction (NSP)**:
   - **Limitation**: The NSP task is relatively simple and may not capture complex relationships between sentences. It can result in **inefficient training** because BERT often learns shallow patterns that do not contribute to improved performance on certain downstream tasks.
   - **False Negatives**: The random pairing of sentences for negative samples can introduce noise, as some randomly paired sentences may be loosely related, confusing the model.

**Impact on Downstream Performance:**
- The limitations of MLM and NSP can lead to **suboptimal contextual understanding** and reduced performance in tasks requiring fine-grained sentence relationships, like **natural language inference** or **paragraph-level comprehension**.
- The mismatch between the pre-training phase (due to the presence of `[MASK]` tokens) and fine-tuning phase (where real tokens are present) may cause challenges in real-world applications.

BERT was originally pre-trained using two primary objectives, each applied to a specific proportion of the training data:

1. **Masked Language Modeling (MLM)**:
   - **Description**: In MLM, 15% of tokens in each input sequence are randomly selected for masking. The selected tokens are processed as follows:
     - **80%** are replaced with the `[MASK]` token.
     - **10%** are replaced with a random token.
     - **10%** remain unchanged.
   - **Data Proportion**: MLM is applied to **100% of the input sequences** during pre-training, making it the dominant objective for learning contextual representations.
   - **Contextual Learning**: By masking tokens, MLM enables BERT to learn bidirectional context by considering both preceding and succeeding tokens.
   - **Limitation**: The model sees the `[MASK]` token only during pre-training, which introduces a discrepancy during fine-tuning, potentially affecting real-world performance.

2. **Next Sentence Prediction (NSP)**:
   - **Description**: BERT receives two sentences and predicts whether the second sentence logically follows the first.
     - **Positive pairs (50%)**: The second sentence follows the first one in the original text.
     - **Negative pairs (50%)**: The second sentence is randomly selected from another part of the corpus.
   - **Data Proportion**: NSP is applied to **50% of the input data** during pre-training, with an equal distribution of positive and negative sentence pairs.
   - **Sentence-Level Learning**: NSP helps BERT capture inter-sentence relationships, which is useful for tasks like question answering and text entailment.
   - **Limitation**: NSP's simplicity can introduce noise and fails to capture complex sentence relationships, reducing its effectiveness in understanding more nuanced contexts.

**Variants of BERT and Their Adjusted Objectives with Proportions:**

1. **RoBERTa**:
   - **Dynamic MLM**: RoBERTa uses only MLM, applied to **100% of the data**, with a dynamic masking strategy that changes the mask pattern across epochs.
   - **NSP Removed**: No data is used for NSP in RoBERTa, allowing the model to focus entirely on more robust MLM, resulting in better performance and generalization.

2. **DistilBERT**:
   - **MLM with Distillation**: DistilBERT retains MLM, which is applied to **100% of the data**. It also uses **knowledge distillation** to learn from a larger teacher model (BERT).
   - **Simplified Training**: There is no NSP, and the distillation process emphasizes efficient learning of the teacher’s outputs.

3. **ALBERT**:
   - **MLM and SOP**: ALBERT keeps MLM, applied to **100% of the data**, while replacing NSP with the **Sentence Order Prediction (SOP)** task.
   - **SOP Task**: SOP is applied to **50% of the input data**, similar to NSP in BERT, but the focus is on predicting whether two consecutive text segments are in order, leading to better sentence coherence understanding.

4. **TaBERT**:
   - **Hybrid MLM**: TaBERT applies MLM to **100% of the textual data** and includes **column prediction** objectives for handling tabular data, adapting MLM for structured data tasks.
   - **Adapted Pre-training**: The objectives are tailored for tabular data, making them less generalizable to purely text-based tasks.

5. **mBERT (Multilingual BERT)**:
   - **Multilingual MLM**: mBERT extends MLM to multiple languages, applying it to **100% of the data** across different languages.
   - **No Change to NSP**: mBERT still retains NSP, applied to **50% of the data** with positive and negative sentence pairs, similar to BERT.

By adjusting the proportions and types of pre-training objectives, these BERT variants address limitations in **contextual consistency**, **sentence-level coherence**, and **multilingual understanding**, leading to improved performance in specific NLP tasks.

---

#### Handling Long Sequences

BERT has a maximum input length limitation due to its positional embeddings in self-attention. Examine the most used techniques of how to extend BERT for longer sequences and their trade-offs.

BERT’s maximum input length is typically **512 tokens**, limited by its **positional embeddings** in the self-attention mechanism. To process longer sequences, several techniques have been developed to extend BERT’s capability while managing trade-offs between performance, efficiency, and complexity.

1. **Longformer**
   - **Technique**: 
     - Introduces a combination of **local** and **global attention** patterns instead of full self-attention.
     - Local attention is applied within a fixed window around each token, while global attention is given to certain tokens, like [CLS], to maintain global context.
   - **Trade-offs**:
     - **Efficiency**: Reduces computational cost from quadratic to linear complexity, making it feasible for longer texts.
     - **Performance**: While it captures local dependencies effectively, some global contextual understanding may be lost compared to full attention.

2. **BigBird**
   - **Technique**:
     - Combines **sparse attention patterns** with random, windowed, and global attention, making it suitable for sequences of over 4,096 tokens.
     - It adds random connections to improve information flow, enabling better global understanding.
   - **Trade-offs**:
     - **Efficiency**: Achieves linear scaling in self-attention, making it efficient for long documents.
     - **Performance**: The random attention component can introduce noise, which might affect precision on tasks that require fine-grained sentence relationships.

3. **Transformer-XL**
   - **Technique**: 
     - Extends BERT by introducing **segment-level recurrence** and **relative positional embeddings**.
     - Keeps hidden states from previous segments, allowing context to be preserved across longer sequences.
   - **Trade-offs**:
     - **Efficiency**: Adds memory overhead due to recurrence, but allows for handling longer dependencies.
     - **Performance**: Recurrence helps maintain context better than local attention alone, but introduces complexity in training and inference.

4. **Reformer**
   - **Technique**:
     - Uses **locality-sensitive hashing (LSH)** to replace the original self-attention mechanism, reducing its complexity from quadratic to logarithmic.
     - LSH clusters similar tokens together, computing attention only within these clusters.
   - **Trade-offs**:
     - **Efficiency**: Extremely efficient, with significant reduction in computational cost, making it capable of processing very long sequences.
     - **Performance**: While efficient, the clustering-based attention may overlook important distant dependencies, affecting tasks that require global context understanding.

5. **ETC (Extended Transformer Construction)**
   - **Technique**:
     - Separates input into a **global memory component** and a **local attention component**, where global memory tokens can attend to all tokens while local tokens attend to nearby tokens.
   - **Trade-offs**:
     - **Efficiency**: Reduces computation and memory requirements, making it feasible for longer inputs.
     - **Performance**: Balances local and global context, but may require task-specific tuning of global memory size to optimize performance.

6. **Hierarchical Attention Networks (HANs)**
   - **Technique**:
     - Processes long texts in a **hierarchical manner**: first by encoding individual sentences, then by encoding relationships between sentences.
   - **Trade-offs**:
     - **Efficiency**: More efficient than flat models, as it breaks long sequences into manageable chunks.
     - **Performance**: Maintains sentence-level coherence well but can lose finer token-level details across very long sequences.

---

#### RoBERTa's Training Improvements

RoBERTa modified BERT's pre-training approach by removing NSP and training on longer sequences with larger batch sizes. Analyze how each of these changes contributes to performance gains. What does this suggest about the importance of training configurations?

---

#### ALBERT's Parameter Reduction

ALBERT reduces the number of parameters through cross-layer parameter sharing and factorized embedding parameterization. Explain how these techniques work and evaluate their impact on model scalability and performance.

---

#### DistilBERT and Knowledge Distillation

Discuss the process of knowledge distillation as applied in DistilBERT. How does DistilBERT achieve a balance between model size reduction and performance retention? Analyze the trade-offs involved in distillation.

---

#### Adapters for Parameter-efficient Fine-tuning

Explain how adapter modules can be integrated into BERT for parameter-efficient fine-tuning. Compare this approach to full model fine-tuning in terms of computational efficiency and performance on low-resource tasks.

---

#### Domain-specific BERT Models

Examine the process of creating domain-specific BERT models like BioBERT or SciBERT. What adaptations are made in terms of vocabulary and pre-training data? Evaluate how these changes enhance performance on domain-specific tasks.
