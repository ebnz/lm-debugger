import React, {useState} from "react";
import styled from "styled-components";
import {Divider, Tag, Button, Table, Tooltip, Menu, InputNumber, Space, Collapse} from 'antd';
import {LayerPrediction, ValueId} from "../types/dataModel";
import {LabelContainer, PredictionContainer} from "./LabelContainer";
import {TextInput} from "./TextInput";
import {QuestionCircleOutlined} from "@ant-design/icons";
import {text} from "node:stream/consumers";


interface Props {
  layer: LayerPrediction;
  onAnalyze: (valueId: ValueId) => void;
  onCopy: (valueId: any) => void;
}


export function Layer(props: Props): JSX.Element {
  let {
    predictions_before,
    predictions_after
  } = props.layer;

  let [textIntervention, setTextIntervention] = useState(props.layer.text_inputs);
  let [layerIndex, setLayerIndex] = useState(props.layer.layer);

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

  let layer_name = "Layer ?";
  if (props.layer.layer >= 0) {
    layer_name = `Layer ${props.layer.layer}`;
  } else if (props.layer.layer === -1) {
    layer_name = "Metric";
  } else if (props.layer.layer === -2) {
    layer_name = "All Layers";
  }

  function add_intervention_button_disabled(text_fields: object) {
    if (Object.values(text_fields).some(value => value === "")) {
      return true;
    } else if (!Object(text_fields)["prompt"].includes("{}")) {
      return true;
    }
    return false;
  }

  const table_columns = [
    {
      title: 'Descriptor',
      dataIndex: 'descriptor',
      key: 'descriptor'
    },
    {
      title: 'Data',
      dataIndex: 'datafield',
      key: 'datafield',
      width: '20%'
    }
  ];

  const { Panel } = Collapse;

  return (
      <LayerLayout>
        <Collapse>
          <Panel key={1} header={
            <>
              <LayerTag color="#a55397">
                {
                  props.layer.layer >= 0 && props.layer.type !== "LMDebuggerIntervention" ?
                    <InputNumber
                      min={props.layer.min_layer}
                      max={props.layer.max_layer}
                      defaultValue={layerIndex}
                      prefix={<span onClick={(e) => e.stopPropagation()}>Layer</span>}
                      size="small"
                      controls={false}
                      precision={0}
                      onChange={(newVal) => setLayerIndex(newVal)}
                      onClick={(e) => e.stopPropagation()}
                      onFocus={(e) => e.stopPropagation()}
                    /> :
                  layer_name}
              </LayerTag>
              <LayerTag color="#a55397">Type {props.layer.type}</LayerTag>
              <Tooltip title={props.layer.docstring}><QuestionCircleOutlined/></Tooltip>
            </>
          }>
            {
              typeof predictions_before !== "undefined" &&
                <MyDivider
                  orientation="left"
                  orientationMargin="15px"
                >Before:</MyDivider>
            }
            {
              typeof predictions_before !== "undefined" &&
                <PredictionContainer
                  predictions={predictions_before}
                />
            }

            {
              typeof props.layer.significant_values !== "undefined" &&
                <MyDivider
                  orientation="left"
                  orientationMargin="15px"
                >Dominant sub-updates:</MyDivider>
            }
            {
              typeof props.layer.significant_values !== "undefined" &&
                <LabelContainer
                  valueLabels={props.layer.significant_values}
                  type={props.layer.type}
                  onAnaylze={props.onAnalyze}
                  onCopy={props.onCopy}
                />
            }

            {
              typeof predictions_after !== "undefined" &&
                <MyDivider
                  orientation="left"
                  orientationMargin="15px"
                >After:</MyDivider>
            }
            {
              typeof predictions_after !== "undefined" &&
                <PredictionContainer
                  predictions={predictions_after}
                />
            }

            {
              typeof props.layer.text_outputs !== "undefined" && <ContentLayout>
                <Table
                  className={"tight-table"}
                  style={{ marginTop: 0, marginBottom: 0 }}
                  dataSource={text_outputs_table_data}
                  columns={table_columns}
                  size="small"
                  pagination={textOutputsMap.size > 5 ? {position: ["bottomRight"]} : false}
                />
              </ContentLayout>
            }

            {
              typeof props.layer.text_inputs !== "undefined" && <TextInputLayout>
                <TextInput
                  textIntervention={textIntervention}
                  setTextIntervention={setTextIntervention}
                ></TextInput>
                <Button
                  disabled={add_intervention_button_disabled(textIntervention)}
                  onClick={(e) => {props.onCopy(
                  {text_inputs: textIntervention,
                  type: props.layer.type,
                  layer: layerIndex,
                  dim: Date.now()}
                  )}}>Add as Intervention</Button>
              </TextInputLayout>
            }
          </Panel>
        </Collapse>



        {/* </SignificantValuesDiv> */}
      </LayerLayout>
  )
}

const ContentLayout = styled.div`
  padding: 8px;
  margin: 0 auto; /* centers the component */
  width: calc(100% - 100px);
`;

const TextInputLayout = styled.div`
  padding: 12px;
  margin: 0 auto; /* centers the component */
  //width: calc(100% - 100px);
  max-width: 90%;
`;

const LayerLayout = styled.div`
  padding: 2px;
  margin: 4px;
  border: 1px #757373c5 solid;
  border-radius: 5px;
  //width: calc(100% - 100px);
  max-width: 90%;
  width: 100%;
  height: fit-content;
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
  font-size: 15pt;
  padding: 2px 30px;
  height: 30px;
`;
