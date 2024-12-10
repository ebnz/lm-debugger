import React from "react";
import styled from "styled-components";
import { Divider, Tag } from 'antd';
import { LayerPrediction, ValueId } from "../types/dataModel";
import {LabelContainer, PredictionContainer} from "./LabelContainer";


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
        <LayerTag color="#a55397">Type {props.layer.type}</LayerTag>
        {typeof predictions_before !== "undefined" && <MyDivider orientation="left" orientationMargin="15px">Before:</MyDivider>}
        {typeof predictions_before !== "undefined" && <PredictionContainer predictions={predictions_before}/>}

        {typeof props.layer.significant_values !== "undefined" && <MyDivider orientation="left" orientationMargin="15px">Dominant sub-updates:</MyDivider>}
        {typeof props.layer.significant_values !== "undefined" && <LabelContainer
          valueLabels={props.layer.significant_values}
          type={props.layer.type}
          onAnaylze={props.onAnalyze}
          onCopy={props.onCopy}
        />}
        {typeof predictions_after !== "undefined" && <MyDivider orientation="left" orientationMargin="15px">After:</MyDivider>}
        {typeof predictions_after !== "undefined" && <PredictionContainer predictions={predictions_after}/>}


        {/* </SignificantValuesDiv> */}
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
