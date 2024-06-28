import React from "react";
import styled from "styled-components";
import { Divider, Tag } from 'antd';
import {AutoEncoderResponse, LayerPrediction, ValueId} from "../types/dataModel";
import {LabelContainer, PredictionContainer, SparseFeatureContainer} from "./LabelContainer";


interface Props {
  layer: LayerPrediction;
  onAnalyze: (valueId: ValueId) => void;
  onCopy: (valueId: ValueId) => void;
}


function Layer(props: Props): JSX.Element {
  let {
    predictions_before,
    predictions_after
  } = props.layer;


  return (
      <LayerLayout>
        <LayerTag color="#a55397">Layer {props.layer.layer}</LayerTag>
        <MyDivider orientation="left" orientationMargin="15px">Before:</MyDivider>
        <PredictionContainer predictions={predictions_before}/>
        
        <MyDivider orientation="left"  orientationMargin="15px">Dominant sub-updates:</MyDivider>
        <LabelContainer 
            valueLabels={props.layer.significant_values}
            onAnaylze={props.onAnalyze}
            onCopy={props.onCopy}
        />
        <MyDivider orientation="left" orientationMargin="15px">After:</MyDivider>
        <PredictionContainer predictions={predictions_after}/>


        {/* </SignificantValuesDiv> */}
      </LayerLayout>
  )
}

interface AutoEncoderProps {
  response: AutoEncoderResponse;
  onAnalyze: (valueId: ValueId) => void;
  onCopy: (valueId: ValueId) => void;
}

function AutoEncoderLayer(props: AutoEncoderProps): JSX.Element {
    let {
        response,
        onAnalyze,
        onCopy
    } = props;
    function getUserReadableLayerType(layer_type: string) {
        const mapping: Map<string, string> = new Map([
            ["self_attn", "Self-Attention AutoEncoder"],
            ["mlp", "MLP AutoEncoder"]
        ]);

        return mapping.has(layer_type) ? mapping.get(layer_type) : "AutoEncoder";
    }

    return(
      <LayerLayout>
        <LayerTag color="#53a58a">{getUserReadableLayerType(response.autoencoder_layer_type)}</LayerTag>
        <MyDivider orientation="left" orientationMargin="15px">Most activating AutoEncoder-Neurons: </MyDivider>
        <SparseFeatureContainer autoencoder_result={response}></SparseFeatureContainer>
      </LayerLayout>
    )
}

const LayerLayout = styled.div`

  padding: 10px;
  margin: 2px;
  border: 1px #757373c5 solid;
  border-radius: 5px;

`;

const MyDivider = styled(Divider)`
  &.ant-divider{
    margin: 5px 0;
  }
  & .ant-divider-inner-text {
    font-size: 12pt;
  }
`;

const LayerTag = styled(Tag)`
  font-weight: bold;
  font-size: 12pt;
  padding: 2px 30px;
`;

export const MemoLayer = React.memo(Layer);
export const MemoAutoEncoderLayer = React.memo(AutoEncoderLayer);
