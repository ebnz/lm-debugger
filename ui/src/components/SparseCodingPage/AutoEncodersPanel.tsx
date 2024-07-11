import React from "react";
import {AutoEncoderResponse, ValueId} from "../../types/dataModel";
import {Empty, Card, Alert, Spin, Divider, Tag, Tooltip, Button, Select} from "antd";
import type {SelectProps} from "antd";
import styled from "styled-components";
import {SparseFeatureContainer} from "../LabelContainer";

interface TokenDescriptorProps {
    text: string,
    activations: Array<number>,
    color: Array<number>    //[0-255, 0-255, 0-255] (RGB)
}
function TokenDescriptor(props: TokenDescriptorProps): JSX.Element {
    const {text, activations, color} = props;
    let alpha = 0.0;
    if (activations[0] > 2 || activations[1] > 2 || activations[2] > 2) {
        alpha = 1.0;
    }
    const rgba_string = "rgb(" + color[0] + ", " + color[1] + ", " + color[2] + ", " + alpha + ")";

    if (text === "<0x0A>") {
        return (
            <>
                <Tooltip title={Math.max(...activations).toFixed(1)}>
                    <text style={{backgroundColor: rgba_string}}>\n</text>
                </Tooltip>
                <br/>
            </>
        )
    }

    return (<>
        <Tooltip title={Math.max(...activations).toFixed(1)}>
            <text style={{backgroundColor: rgba_string}}>{text.replace("▁", " ")}</text>
        </Tooltip>
    </>)
}

interface AEPanelProps {
    isLoading: boolean,
    errorMessage?: string,
    autoencoder_results: Array<AutoEncoderResponse>;
    autoencoderFeatures: Array<number>;
    handleAutoencoderFeaturesChange: Function;
}

function AutoEncodersPanel(props: AEPanelProps): JSX.Element {
  const {
    isLoading,
    errorMessage,
    autoencoder_results,
    autoencoderFeatures,
    handleAutoencoderFeaturesChange
  } = props;

  let contentRender: React.ReactNode = [];
  if (isLoading) {
    contentRender = <Spin style={{margin: "auto auto"}} tip="Loading Feature Activations" />;
  } else if (errorMessage !== undefined) {
    contentRender = <Alert type="error">{errorMessage}</Alert>
  } else if (autoencoder_results.length === 0){
    contentRender = <Empty description="Enter a prompt and click Run to see Feature Activations."/>
  } else {
    let contentRenderArray = [];


    for (let index = 0; index < autoencoder_results[0].tokens_as_string.length; index++) {
        let color = [0.0, 0.0, 0.0];

        color[0] += autoencoder_results.length >= 1 ? autoencoder_results[0].neuron_activations[index] * 255 / 10 : 0.0;
        color[1] += autoencoder_results.length >= 2 ? autoencoder_results[1].neuron_activations[index] * 255 / 10 : 0.0;
        color[2] += autoencoder_results.length >= 3 ? autoencoder_results[2].neuron_activations[index] * 255 / 10 : 0.0;

        const activations = [
            autoencoder_results.length >= 1 ? autoencoder_results[0].neuron_activations[index] : 0.0,
            autoencoder_results.length >= 2 ? autoencoder_results[1].neuron_activations[index] : 0.0,
            autoencoder_results.length >= 3 ? autoencoder_results[2].neuron_activations[index] : 0.0
        ];

        contentRenderArray.push(
            <TokenDescriptor key={index} text={autoencoder_results[0].tokens_as_string[index]} activations={activations} color={color}></TokenDescriptor>
        );
    }
    contentRender = contentRenderArray;
  }

  function handleFeaturesChange(values: Array<string>) {
      if (values.length > 3) {
          //ToDo: Implement behavior
          return;
      }
      const numberValues = values.map(Number);
      for (let item of numberValues) {
          if (isNaN(item)) {
              //ToDo: Open e.g. Toast
              handleAutoencoderFeaturesChange(autoencoderFeatures);
              return;
          }
      }
      handleAutoencoderFeaturesChange(numberValues.map(Math.floor));
  }

  type TagRender = SelectProps['tagRender'];

  const tagRender: TagRender = (props) => {
  const { label, value, closable, onClose } = props;
  const COLORS = ["red", "green", "blue"];

  return (
    <Tag
      color={COLORS[autoencoderFeatures.indexOf(parseFloat(value))]}
      closable={closable}
      onClose={onClose}
    >
      {label}
    </Tag>
  );
};

  return (
      <MainLayout>
          <Select
            mode="tags"
            style={{ width: '100%' }}
            value={autoencoderFeatures.map(String)}
            onChange={handleFeaturesChange}
            tagRender={tagRender}
          />
          <Divider/>
          {contentRender}
      </MainLayout>
  );
}

const MainLayout = styled(Card).attrs({
    size: "small"
})`
    width: 100%;
    height: 100%;

  &.ant-card .ant-card-body {
    height: calc(100vh - 236px);
    overflow-x: hidden;
    overflow-y: auto;
    padding: 2px;
  }
`;

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

export default AutoEncodersPanel;