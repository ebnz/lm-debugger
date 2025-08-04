{
  server_files_dir:: "server_files/",
  model_name:"codellama/CodeLlama-7b-Instruct-hf",
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
  elastic_index:"codellama_7b_instruct_projections_docs",
  elastic_projections_path:$.server_files_dir + "values_logits_top_" + $.top_k_for_elastic + ".pkl",
  elastic_api_key: "VGhlIGNha2UgaXMgYSBsaWU=",

  layer_mappings: {
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

  sae_paths: [
    "autoencoders/my_sae.pt"
  ],
  autoencoder_device: "cuda:1",
  sae_active_coeff: 100,

  rome_paths: [
    "config_files/ROME/codellama_CodeLlama-7b-Instruct-hf.json"
  ],

  easy_edit_hparams_path: "config_files/ee_hparams",
}
