{
  "dataset_reader": {
    "type": "gen_mcqa",
    "max_dataset_size": 15000,
    "max_pieces": 90,
    "shuffle": true,
    "use_bert": false,
    "use_xlnet": true,
    "random_seed": 1,
    "add_prefix": false,
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_local",
        "model_name":  "xlnet-large-cased",
        "do_lowercase": true,
        "need_separator": true,
        "use_xlnet": true,
        "use_bos_as_padding": true, // This is needed to use <|endoftext|> token instead of a new <pad> token
        "padding_on_right": false
      }
    }
  },
  "train_data_path":  "/home_dir/allen/commonsense/train_rand_split.jsonl",
  "validation_data_path": "/home_dir/allen//commonsense/dev_rand_split.jsonl",
  "model": {
    "type": "general_gen_cls",
    "model_name": "xlnet-large-cased",
    "k": 1,
    "train_lm_generator": false,
    "use_kld_loss": false,
    "use_repetition_loss": false,
    "anneal_repetition_loss": false,
    "generate_until_dot": true,
    "use_cls": true,
    "use_similarity": true,
    "use_similarity_btw_question_and_answers": false,
    "zero_generated_out": false,
    "dropout": 0.1,
    "add_cls_after_epoch_num": 3,
    "use_straight_through_gumbel_softmax": true,
    "anneal_temperature": true,
    "output_several_results_on_every_step": true,
    "results_each_step": 5,
     "load_weights": false,
//        "initializer": [[".*(gen|cls).*",
//                      {
//                        "type": "pretrained",
//                        "weights_file_path": "/specific/netapp5/joberant/home/veronical/allen/commonsense/final_logs/xlnet_large_sim_top5_for_real_gs_no_kld_no_rep_cls_after_3/best.th",
//                        "cuda_device": 0
//                      }],]

  },
  "iterator": {
    "type": "basic",
    "track_epoch": true,
    "batch_size": 1
  },
  "trainer": {
    "num_epochs": 25,
    "patience": 10,
    "validation_metric": "+accuracy",
    "gradient_accumulation_steps": 8,
    "num_serialized_models_to_keep": 1,
    "cuda_device": [
      0
    ],
    "optimizer": {
      "type": "bert_adam",
      "lr": 0.000002,
//      "warmup": 0.2
    }
  }
}