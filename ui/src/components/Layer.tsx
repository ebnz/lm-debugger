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

        {typeof props.layer.text_outputs !== "undefined" && <MyDivider orientation="left" orientationMargin="15px">Data:</MyDivider>}
        {typeof props.layer.text_outputs !== "undefined" && <Table dataSource={text_outputs_table_data} columns={columns} />}

        {typeof props.layer.text_inputs !== "undefined" && <MyDivider orientation="left" orientationMargin="15px">Text Inputs:</MyDivider>}
        {typeof props.layer.text_inputs !== "undefined" && <TextInput textIntervention={textIntervention} setTextIntervention={setTextIntervention}></TextInput>}
        {typeof props.layer.text_inputs !== "undefined" && <MyDivider orientation="left" orientationMargin="15px">Actions:</MyDivider>}
        {typeof props.layer.text_inputs !== "undefined" && <Button onClick={(e) => {props.onCopy(
            {text_inputs: textIntervention,
                type: props.layer.type,
                layer: props.layer.layer,
                dim: textIntervention["subject"] + textIntervention["target"] + textIntervention["prompt"]}
        )}}>Add as Intervention</Button>}


        {/* </SignificantValuesDiv> */}
      </LayerLayout>
  )
}

const LayerLayout = styled.div`

  padding: 10px;
  margin: 2px;
  border: 1px #757373c5 solid;
  border-radius: 5px;
  width: calc(100% - 100px);
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
