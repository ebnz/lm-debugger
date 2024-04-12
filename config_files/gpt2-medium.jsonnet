{
  server_files_dir:: "SERVER_FILES_PATH",
  model_name:"gpt2-medium",
  device:"cuda",
  server_ip: "129.206.61.57",
  server_port: 8000,
  elastic_ip: "129.206.61.57",
  elastic_port: 9200,
  react_ip: "129.206.61.57",
  react_port: 3000,
  streamlit_ip: "129.206.61.57",
  streamlit_port: 8501,
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
