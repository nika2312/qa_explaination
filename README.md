# qa_explaination
Official Github repo for the paper "Explaining Question Answering Models through Controlled Text Generation"

This repo is basically an allennlp package, so allennlp installation is required, as well as some adjustments to its trainer.py file to allow gradient accumulation.

The current configuration is set to reproduce the TOP-K=3 model, with the similarity and LM-based classifiers trained jointly on CommonsenseQA.

To run training:
`allennlp train ./qa_explanation/configs/config.jsonnet -s YOUR_OUTPUT_LOCATION --include-package qa_explanation`
