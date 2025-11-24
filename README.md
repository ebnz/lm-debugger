[//]: # (# LM-Debugger 🔎 )
### 
<img width="30%" src="img/knowledge_editor_logo.png" />

LM-Debugger++ is an open-source interactive tool for inspection and intervention in transformer-based language models. LM-Debugger++ is an extension of the original LM-Debugger, originally developed by [Geva et al](https://arxiv.org/abs/2204.12130).
This repository includes the code and links for data files required for running LM-Debugger++ over CodeLlama Models by Facebook. Adapting this tool to other models only requires changing the backend API (see details below). 
Contributions our welcome!


An online demo of the original LM-Debugger is available at: 
- GPT2 Medium: https://lm-debugger.apps.allenai.org/
- GPT2 Large: https://lm-debugger-l.apps.allenai.org/


[<img width="70%" src="https://user-images.githubusercontent.com/18243390/164968806-6e56f993-8cca-4c27-9e27-adaaa6ebc904.png"/>](http://www.youtube.com/watch?v=5D_GiJv7O-M "LM-Debugger demonstration")

<p align="center"><img width="30%" src="img/img.png" /></p>


### ⚙️ Requirements

LM-Debugger++ has two main views for (a) debugging and intervention in model predictions, and (b) exploration of information encoded in the model's feed-forward layers.

The tool runs in a React and python environment with Flask and Streamlit installed. In addition, the exploration view uses an Elasticsearch index. To set up the environment, please follow the steps below:

1. Clone this repository:
   ```bash
   git clone https://github.com/ebnz/lm-debugger
   cd lm-debugger
   ```
2. Create a Python 3.10 environment, and install the following dependencies via Conda

3. Install [sparse-autoencoders](https://github.com/ebnz/sparse-autoencoders) with Conda Develop

4. Install [Yarn](https://yarnpkg.com/) and [NVM](https://github.com/nvm-sh/nvm), and set up the React environment:
   ```bash
   cd ui
   nvm install
   yarn install
   cd ..
   ```

5. Install [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html) and make sure that the service is up.



### 🔎 Running LM-Debugger

#### Creating a Configuration File 
LM-Debugger executes one model at a time, based on a given configuration file. 
This file's config fields can be seen in the [Docs](https://ebnz.github.io/lm-debugger/).

#### Creating an Elasticsearch Index
The keyword search functionality in the exploration view is powered by an Elasticsearch index that stores the projections of feed-forward parameter vectors from the entire network. To create this index, run:
```bash
python es_index/index_value_projections_docs.py \
--config_path CONFIG_PATH
```

#### Creating a Sparse Autoencoder
A Sparse Autoencoder Model for use with LM-Debugger++ can be trained with the accompanying Library [sparse-autoencoders](https://github.com/ebnz/sparse-autoencoders). Explanation in the Docs. 

#### Creating a ROME-Instance
A ROME Instance can be defined like the ones in [config_files/ROME](config_files/ROME). Additionally, this file must be registered in the LM-Debugger++ Configuration.

#### Executing LM-Debugger

To run LM-Debugger:
```bash
bash start.sh CONFIG_PATH
```

------------------------------
In case you are interested in _running only one of the two views of LM-Debugger_, this can be done as follows:

1. To run the Flask server (needed for the prediction view):
   ```bash
   python flask_server/app.py --config_path CONFIG_PATH
   ```

2. To run the prediction view:
   ```bash
   python ui/src/convert2runConfig.py --config_path CONFIG_PATH
   cd ui
   yarn start
   ```

3. To run the exploration view:
   ```bash
   streamlit run streamlit/exploration.py -- --config_path CONFIG_PATH
   ```


### Citation
Please cite as:
```bibtex
@article{geva2022lmdebugger,
  title={LM-Debugger: An Interactive Tool for Inspection and Intervention in Transformer-Based Language Models},
  author={Geva, Mor and Caciularu, Avi and Dar, Guy and Roit, Paul and Sadde, Shoval and Shlain, Micah and Tamir, Bar and Goldberg, Yoav},
  journal={arXiv preprint arXiv:2204.12130},
  year={2022}
}
```

Docs to be extended soon!