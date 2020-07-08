# qa_explaination
Official Github repo for the paper ["Explaining Question Answering Models through Text Generation"](https://arxiv.org/abs/2004.05569).

This repo is basically an allennlp package, so allennlp installation is required, as well as some adjustments to its trainer.py file if you want to allow gradient accumulation.

The current configuration is set to reproduce the TOP-K=3 model results on a V100 GPU, with the similarity and LM-based classifiers trained jointly on CommonsenseQA.

To train the model, update the paths in the configs/config.jsonnet file.

If you want to use gradient accumulation, you can use the changes I added in my trainer.py file (allennlp version 0.9.0) and incorporate them in your respective /allennlp/training/trainer.py file. 

In any case, delete the trainer.py file from the current package afterwards.

Then run:
`allennlp train ./qa_explanation/configs/config.jsonnet -s YOUR_OUTPUT_LOCATION --include-package qa_explanation`
