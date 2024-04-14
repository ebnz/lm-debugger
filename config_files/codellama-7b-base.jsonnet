{
  server_files_dir:: SERVER_FILES_PATH,
  model_name:"codellama/CodeLlama-7b-hf",
  device:"cpu",
  server_ip:SERVER_IP,
  server_port:SERVER_PORT,
  elastic_ip:ELASTIC_IP,
  elastic_port:ELASTIC_PORT,
  react_ip:REACT_IP,
  react_port:REACT_PORT,
  streamlit_ip:STREAMLIT_IP,
  streamlit_port:STREAMLIT_PORT,
  top_k_tokens_for_ui:10,
  top_k_for_elastic:50,
  create_cluster_files:false,
  num_clusters:3000,
  num_layers:24,
  elastic_index:$.model_name+"_projections_docs",
  elastic_projections_path:$.server_files_dir + "values_logits_" + $.model_name +"_top_"+$.top_k_for_elastic + ".pkl",
  streamlit_cluster_to_value_file_path: $.server_files_dir + "cluster_to_value_" + $.model_name +"_num_clusters_"+$.num_clusters + ".pkl",
  streamlit_value_to_cluster_file_path: $.server_files_dir + "value_to_cluster_" + $.model_name +"_num_clusters_"+$.num_clusters + ".pkl",
}
