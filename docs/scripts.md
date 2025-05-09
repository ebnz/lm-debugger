# Training Scripts

#### Generate Tokenized Dataset

This Script generates a Tokenized Dataset from a HuggingFace-Dataset and stores it on the disk. 

Parameters: 

* `--save_path SAVE_PATH`: Path to save-directory for the generated Dataset
* `--min_tokens MIN_TOKENS`: Number of minimum Tokens one Context has to contain in order to be used in the generated Dataset
* `--num_processes NUM_PROCESSES`: Number of concurrent Processes for downloading Contexts
* `--tokenizer_name TOKENIZER_NAME`: HuggingFace-Name for the Tokenizer for tokenizing the Contexts. Currently only works for CodeLlama-Tokenizers
* `--dataset_name DATASET_NAME`: HuggingFace-Name for Dataset to load data from
* `num_files`: Number of Files to generate. If Multiprocessing is used: Has to be multiple of num_processes
* `num_samples_per_file`: Number of tokenized Contexts per File

Example: 

```bash
python3 generate_tokenized_dataset.py --save_path ~/llama_tokenized_dataset --min_tokens 256 --num_processes 16 16 10000
```

Generates a Tokenized Dataset of Samples, containing at least 256 Tokens with 16 Processes. Saves this Dataset into 16 Files, each containing 10000 Tokenized Contexts. 

#### Train Autoencoder

This Script trains a Sparse Autoencoder Model. 

Parameters: 

* `--num_batches NUM_BATCHES`: Amount of Batches (AutoEncoder-sized) used to train the AutoEncoder on
* `--dataset_path DATASET_PATH`: Path to the Pretokenized Dataset
* `--save_path SAVE_PATH`: Path to save the trained AutoEncoders to
* `--model_name MODEL_NAME`: HuggingFace-Name for the Model for obtaining Activations. Currently only works for CodeLlama-Models
* `--batch_size_llm BATCH_SIZE_LLM`: Batch Size used to obtain Model Activations from the LLM
* `--batch_size_autoencoder BATCH_SIZE_AUTOENCODER`: Batch Size used to train the AutoEncoder
* `--num_tokens NUM_TOKENS`: Amount of Tokens used to train AutoEncoder
* `--device_llm DEVICE_LLM`: Device to load the LLM to
* `--device_autoencoder DEVICE_AUTOENCODER`: Device to load the AutoEncoder to
* `--learning_rate LEARNING_RATE`: Learning-Rate for AutoEncoder-Training.
* `--l1_coefficient L1_COEFFICIENT`: L1-Coefficient or Sparsity-Coefficient for AutoEncoder-Training.
* `--act_vec_size ACT_VEC_SIZE`: Size of the Activation-Vector inputted to the AutoEncoder.
* `--dict_vec_size DICT_VEC_SIZE`: Size of the Dictionary-Vector produced by the AutoEncoder.
* `--batches_between_ckpt BATCHES_BETWEEN_CKPT`: Number of Batches to train the AutoEncoder, before the Model is saved as a Checkpoint-File and a Feature-Frequencies-Image is generated.
* `--num_batches_preload NUM_BATCHES_PRELOAD`: Buffer-Size of Activation-Vectors for Training. If Buffer is empty, will be refilled. Larger Buffer results in higher Randomness while Training.
* `--neuron_resampling_method NEURON_RESAMPLING_METHOD`: Strategy for Neuron-Resampling. Currently available: 'replacement', 'anthropic'
* `--neuron_resampling_interval NEURON_RESAMPLING_INTERVAL`: Amount of Batches (AutoEncoder-sized) to train, until Neuron-Resampling
* `--normalize_dataset`: If activated, all Activation Vectors in the Dataset will be normalized to L2-Norm of sqrt(n) (with n being input dimension of AutoEncoder)
* `--mlp_activations_hookpoint MLP_ACTIVATIONS_HOOKPOINT`: Hookpoint description for MLP-Activations. e.g. model.layers.{}.mlp.act_fn ({} for Layer Index)
* `--mlp_sublayer_hookpoint MLP_SUBLAYER_HOOKPOINT`: Hookpoint description for MLP-Sublayer. e.g. model.layers.{}.mlp ({} for Layer Index)
* `--attn_sublayer_hookpoint ATTN_SUBLAYER_HOOKPOINT`: Hookpoint description for Attention-Sublayer. e.g. model.layers.{}.self_attn ({} for Layer Index)
* `layer_id`: ID of Layer from which the Activations are obtained
* `layer_type`: Type of Layer from which the Activations are collected. Select 'attn_sublayer', 'mlp_sublayer' or 'mlp_activations'.

Example: 

```bash
python3 train_autoencoder_from_tokens.py --num_batches 50000 --dataset_path ~/tokenized_dataset/ --save_path ~/autoencoders/l19_lr2e-4_spar0.5 --batch_size_llm 16 --batch_size_autoencoder 1024 --num_tokens 64 --device_llm cuda:0 --device_autoencoder cuda:1 --learning_rate 0.0002 --batches_between_ckpt 5000 --num_batches_preload 5000 19 mlp_activations
```

#### Generate Interpretation Samples

This Script samples Activations of the Neurons of the Sparse Autoencoder and generates heuristics on their Activation. 

Parameters: 

* `--dataset_path DATASET_PATH`: Path of Tokenized Dataset to use for Obtaining Interpretation Samples
* `--autoencoder_path AUTOENCODER_PATH`: Path of Autoencoder Model to analyze
* `--save_path SAVE_PATH`: Path to save the Interpretation Samples to
* `--target_model_name TARGET_MODEL_NAME`: Name of Target-Model. Currently, only CodeLlama-Models are supported
* `--target_model_device TARGET_MODEL_DEVICE`: Device of Target Model
* `--autoencoder_device AUTOENCODER_DEVICE`: Device of Autoencoder
* `--log_freq_upper LOG_FREQ_UPPER`: Maximal Log-Feature-Frequency at which a Feature should be interpreted
* `--log_freq_lower LOG_FREQ_LOWER`: Minimal Log-Feature-Frequency at which a Feature should be interpreted
* `num_samples`: Number of Interpretation Samples to obtain

Comment on `log_freq_upper` and `log_freq_lower`: The Training Process of an SAE produces the Model and also Frequency-Histograms. Features, that appear very often/rare might not be of interest to be interpreted and can be excluded from Interpretation (very resource-intensive). The Values of `log_freq_upper` and `log_freq_lower` refer to $\log_{10}(\text{Probability of Activation of Neuron})$. So if `log_freq_upper` is $-0.1$, Features that have a higher probability to activate, than $10^{-0.1}$ are discarded. 

Example: 

```bash
python3 generate_interpretation_samples.py --dataset_path ~/tokenized_dataset/ --autoencoder_path ~/l19_lr2e-4_spar0.5/50000.pt --save_path ~/interp_samples_l19.pt --target_model_device cuda:3 --autoencoder_device cuda:2 10000
```

#### Interpret Autoencoder

This Script interprets the Sparse Autoencoder. **Only startable with Deepspeed!**

Parameters: 

* `--dataset_path DATASET_PATH`: Path of Tokenized Dataset to use for Obtaining Interpretation Samples
* `--interpretation_samples_path INTERPRETATION_SAMPLES_PATH`: Path of the saved Interpretation-Samples
* `--interpretation_model_name INTERPRETATION_MODEL_NAME`: Name of Interpretation-Model. Currently, only CodeLlama-Models are supported
* `--num_gpus NUM_GPUS`: Number of GPUs to use
* `--autoencoder_path AUTOENCODER_PATH`: Path to Autoencoder to interpret
* `--num_interpretation_samples NUM_INTERPRETATION_SAMPLES`: Number of Interpretation Samples to use
* `--num_simulation_samples NUM_SIMULATION_SAMPLES`: Number of Simulation Samples to use
* `--local_rank LOCAL_RANK`: Local Rank
* `--ssl_cert SSL_CERT`: Path to SSL-Cert of ElasticSearch
* `server_address`: Address to ElasticSearch-Server (e.g. https://DOMAIN:PORT)
* `api_key`: API-Key to ElasticSearch-Server

Example: 

```bash
deepspeed interpret_autoencoder_deepspeed.py --dataset_path ~/tokenized_dataset --interpretation_samples_path ~/interp_samples_l19.pt --num_gpus 4 --autoencoder_path ~/l19_lr2e-4_spar0.5/50000.pt 127.0.0.1:1234 API_KEY
```

