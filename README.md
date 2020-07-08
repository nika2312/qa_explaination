# qa_explaination
Official Github repo for the paper ["Explaining Question Answering Models through Text Generation"](https://arxiv.org/abs/2004.05569)

This repo is basically an allennlp package, so allennlp installation is required, as well as some adjustments to its trainer.py file if you want to allow gradient accumulation.

The current configuration is set to reproduce the TOP-K=3 model results on a V100 GPU, with the similarity and LM-based classifiers trained jointly on CommonsenseQA.

To run training:
`allennlp train ./qa_explanation/configs/config.jsonnet -s YOUR_OUTPUT_LOCATION --include-package qa_explanation`
