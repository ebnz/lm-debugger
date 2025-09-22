import React, {useState} from "react";
import styled from "styled-components";
import {Divider, Tag, Button, Table} from 'antd';
import {LayerPrediction, ValueId} from "../types/dataModel";
import {LabelContainer, PredictionContainer} from "./LabelContainer";
import { TextInput } from "./TextInput";
import {createHash} from "node:crypto";


interface Props {
  layer: LayerPrediction;
  onAnalyze: (valueId: ValueId) => void;
  onCopy: (valueId: any) => void;
}


function Layer(props: Props): JSX.Element {
  let {
    predictions_before,
    predictions_after
  } = props.layer;

  let [textIntervention, setTextIntervention] = useState(props.layer.text_inputs);

  let [textOutputs, setTextOutputs] = useState(props.layer.text_outputs);
  var text_outputs_table_data: object[] = [];

  let textOutputsMap = new Map();
  for (let key of Object.keys(textOutputs || {})) {
    textOutputsMap.set(key, textOutputs[key]);
  }

  let layer_name = "Layer ?";
  if (props.layer.layer >= 0) {
    layer_name = `Layer ${props.layer.layer}`;
  } else if (props.layer.layer === -1) {
    layer_name = "Metric";
  } else if (props.layer.layer === -2) {
    layer_name = "All Layers";
  }

  var index = 0;
  textOutputsMap.forEach((val, key) => {
    text_outputs_table_data.push({
      key: index.toString(),
      descriptor: key,
      datafield: val,
    });
    index += 1;
  })

  const columns = [
  {
    title: 'Descriptor',
    dataIndex: 'descriptor',
    key: 'descriptor',
  },
  {
    title: 'Data',
    dataIndex: 'datafield',
    key: 'datafield',
  }
];

  return (
      <LayerLayout>
        <LayerTag color="#a55397">{layer_name}</LayerTag>
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

        {typeof props.layer.text_outputs !== "undefined" && <ContentLayout><MyDivider orientation="left" orientationMargin="15px">Data:</MyDivider></ContentLayout>}
        {typeof props.layer.text_outputs !== "undefined" && <ContentLayout><Table dataSource={text_outputs_table_data} columns={columns} size="small"/></ContentLayout>}

        {typeof props.layer.text_inputs !== "undefined" && <ContentLayout><MyDivider orientation="left" orientationMargin="15px">Text Inputs:</MyDivider></ContentLayout>}
        {typeof props.layer.text_inputs !== "undefined" && <ContentLayout><TextInput textIntervention={textIntervention} setTextIntervention={setTextIntervention}></TextInput></ContentLayout>}
        {typeof props.layer.text_inputs !== "undefined" && <ContentLayout><MyDivider orientation="left" orientationMargin="15px">Actions:</MyDivider></ContentLayout>}
        {
          typeof props.layer.text_inputs !== "undefined" && <ContentLayout>
            <Button
              disabled={Object.values(textIntervention).some(value => value === "")}
              onClick={(e) => {props.onCopy(
              {text_inputs: textIntervention,
              type: props.layer.type,
              layer: props.layer.layer,
              dim: 0}
            )}}>Add as Intervention</Button>
          </ContentLayout>
        }


        {/* </SignificantValuesDiv> */}
      </LayerLayout>
  )
}

const ContentLayout = styled.div`
  padding: 10px;
  margin: 0 auto; /* centers the component */
  //width: calc(100% - 100px);
  max-width: 90%;
`;

const LayerLayout = styled.div`
  padding: 10px;
  margin: 2px;
  border: 1px #757373c5 solid;
  border-radius: 5px;
  //width: calc(100% - 100px);
  max-width: 70%;
  width: 100%;
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
