BERTective: Language Models and Contextual Information for Deception Detection
------------------------------------------------------------------------------

Reproducibility files for the paper *BERTective: Language Models and Contextual Information for Deception Detection*

pdf at https://aclanthology.org/2021.eacl-main.232.pdf

The fold ``decour.1`` contains the data set.

To replicate the experiments:

- Run ``1_xmls2df.py`` to create the data frame;
- Download the Italian fastText embeddings from https://fasttext.cc/docs/en/crawl-vectors.html and put the file ``cc.it.300.vec`` in the folder ``fasttext``;
- Run ``2_fasttext2embs.py`` to create input files;
- Run ``4_exp200929decour.py`` to run the experiments, setting up appropriately the input folders in the ``args.*`` variables.
The script allows to choose between different models and hyperparameters: enjoy!
- Run ``5_bootstrap.py`` to compute bootstrap sampling.

For any question, please contact me at fornaciari@unibocconi.it
