export interface ValueId {
  layer: number;
  dim: number;
  desc?: string;
}

export interface Intervention extends ValueId{
  coeff: number;
}

export interface PredictionParams {
  prompt: string;
  generate_k: Number,
  interventions?: Array<Intervention>;
}

export interface Prediction {
  token: string;
  score: number;
}

export interface ScoredValue extends ValueId {
  score: number;
}

export interface LayerPrediction {
  layer: number;
  predictions_before: Array<Prediction>;
  predictions_after: Array<Prediction>;
  significant_values: Array<ScoredValue>;
}

export interface NetworkPrediction {
  prompt: string;
  layers: Array<LayerPrediction>;
}

export interface GenerationOutput {
  generate_text: string
}

export interface Value {
  token: string;
  logit: number;
}

export interface ValueInterpretation extends ValueId{
    top_k: Array<Value>
}

//Sparse Coding
export interface AutoEncoderResponse {
  autoencoder_name: string;
  autoencoder_layer_type: string;
  autoencoder_layer_index: number;

  tokens_as_string: Array<string>;
  token_ids: Array<number>;
  neuron_ids: Array<number>;
  interpretations: Array<string>;
  neuron_activations: Array<number>;
}