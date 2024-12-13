import {PredictionParams, NetworkPrediction, ValueId, ValueInterpretation, GenerationOutput} from "../types/dataModel";
import runConfig from "../runConfig.json";
import {toType} from "../types/constants";

export async function generate(params: PredictionParams): Promise<GenerationOutput> {
  const response = await fetch(
    `http://${runConfig.server_ip}:${runConfig.server_port}/generate`,
    {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        prompt: params.prompt,
        generate_k: params.generate_k,
        interventions: params.interventions ?? []
      })
    }
  )

  const responseJson = await response.json();
  return responseJson
}

export async function predict(params: PredictionParams): Promise<NetworkPrediction> {
  const response = await fetch(
    `http://${runConfig.server_ip}:${runConfig.server_port}/get_data`,
    {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        prompt: params.prompt,
        interventions: params.interventions ?? []
      })
    }
  )
  
  const responseJson = await response.json();
  
  // No intervention information is sent when no interventions is provided
  const result = responseJson.intervention ?? responseJson.response;
  // const resultWithNames = addNamesToValues(result)
  return result
  
}

export function getValueNamesFromCookies(): Array<ValueId> {
  // Search cookies
  const cookieData = document.cookie.split(';');
  const newValueIds = cookieData.filter(cookie=> cookie.trim().startsWith("new_name_"))
  .map(cookie => {
    const pair = cookie.split("=")
    const _valueId = pair[0].trim()
    const desc = pair[1].trim()
    let [type_and_layer_str, dim_str] = _valueId.split("D")
    type_and_layer_str = type_and_layer_str.replace("new_name_", "")
    const type = toType.get(type_and_layer_str[0]) ?? "unknown"
    const layer = parseInt(type_and_layer_str.slice(1))
    const dim = parseInt(dim_str)
    const valueId: ValueId = {type, layer, dim, desc}; // ToDo: Check function of this method
    return valueId;
  })
  return newValueIds;
}


export function getValueInterpretation(params: ValueId): [Promise<ValueInterpretation>, () => void] {
  const controller = new AbortController();

  const responsePromise = fetch(
    `http://${runConfig.server_ip}:${runConfig.server_port}/get_projections/type/${params.type}/layer/${params.layer}/dim/${params.dim}`,
    {
      signal: controller.signal
    }
  ).then((r) => {
    return r.json();
  });

  return [responsePromise, () => controller.abort()];
}
