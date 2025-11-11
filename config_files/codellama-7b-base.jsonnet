{
  server_files_dir:: "server_files/",
  model_name:"codellama/CodeLlama-7b-hf",
  device:"cuda:0",
  server_ip: "localhost",
  server_port: 8000,
  elastic_ip: "localhost",
  elastic_port: 9200,
  react_ip: "localhost",
  react_port: 3000,
  streamlit_ip: "localhost",
  streamlit_port: 8501,
  top_k_tokens_for_ui:10,
  top_k_for_elastic:50,
  num_layers:32,
  elastic_index:"codellama_7b_projections_docs",
  elastic_projections_path:$.server_files_dir + "values_logits_top_" + $.top_k_for_elastic + ".pkl",
  elastic_api_key: "VGhlIGNha2UgaXMgYSBsaWU=",

  layer_mappings: {
    token_embedding: "model.embed_tokens",
    mlp_sublayer: "model.layers.{}.mlp",
    attn_sublayer: "model.layers.{}.self_attn",
    mlp_activations: "model.layers.{}.mlp.act_fn",
    mlp_gate_proj: "model.layers.{}.mlp.gate_proj",
    mlp_up_proj: "model.layers.{}.mlp.up_proj",
    mlp_down_proj: "model.layers.{}.mlp.down_proj",
    decoder_input_layernorm: "model.layers.{}.input_layernorm",
    decoder_post_attention_layernorm: "model.layers.{}.post_attention_layernorm",
    post_decoder_norm: "model.norm"
  },

  easy_edit_hparams_path: "config_files/ee_hparams",

  metric_configs: {
    EfficacyMetric: {
      dataset: {
        prompts: ["Elon Musk was born in the city of",
                  "Ian Fleming was born in the city of",
                  "Barack Obama was born in the city of"],
        targets: ["Pretoria", "London", "Honolulu"]
      },
    },
    ExcessiveWeightDeltasMetric: {},
    LocalizationVEditingMetric: {
      # Config for Causal Trace
      samples: 10,
      noise: 0.1,
      window: 10,
      kind: "mlp"
    },
    OutOfDistributionKeysMetric: {
      applicable_intervention_methods: ["ROME", "R-ROME"]
    },
    PerplexityMetric: {}
  }
}
