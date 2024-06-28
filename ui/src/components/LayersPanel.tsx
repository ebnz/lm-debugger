import React from "react";
import {Empty, Card, Alert, Spin } from "antd";
import {AutoEncoderResponse, LayerPrediction, ValueId} from "../types/dataModel";
import styled from "styled-components";
import {MemoLayer, MemoAutoEncoderLayer} from "./Layer"

interface Props {
  layers?: Array<LayerPrediction>;
  setSelectedValueId: (v: ValueId) => void;
  addIntervention: (valueId: ValueId) => void;
  isLoading: boolean;
  errorMessage?: string;
  autoencoder_results: Array<AutoEncoderResponse>;
}

function LayersPanel(props: Props): JSX.Element {
  const {
    layers,
    setSelectedValueId,
    addIntervention,
    isLoading,
    errorMessage,
    autoencoder_results,
  } = props;
  
  let contentRender: React.ReactNode = [];
  if (isLoading) {
    contentRender = <Spin style={{margin: "auto auto"}} tip="Loading prediction" />;
  } else if (errorMessage !== undefined) {
    contentRender = <Alert type="error">{errorMessage}</Alert>
  } else if (layers === undefined){
    contentRender = <Empty description="Run a query to see the predicted layers"/>
  } else {
    let contentRenderArray = [];
    let autoencoder_index = 0;

    for (let item of layers) {
      contentRenderArray.push(
          <MemoLayer
            key={`layer_${item.layer}`}
            layer={item}
            onAnalyze={valueId => setSelectedValueId(valueId)}
            onCopy={addIntervention}
          />
      )
      for (let autoencoder_layer of autoencoder_results) {
        if (item.layer == autoencoder_layer.autoencoder_layer_index) {
        contentRenderArray.push(
            <MemoAutoEncoderLayer
            key={`autoencoder_${autoencoder_layer.autoencoder_layer_index}`}
            response={autoencoder_results[autoencoder_index]}
            onAnalyze={valueId => setSelectedValueId(valueId)}
            onCopy={addIntervention}
          />
        )

        if (autoencoder_index < autoencoder_results.length - 1) {
          autoencoder_index += 1;
        }
      }
      }
    }


    /*contentRender = (
      layers.map((item) => (
         <MemoLayer 
            key={`layer_${item.layer}`}
            layer={item} 
            onAnalyze={valueId => setSelectedValueId(valueId)}
            onCopy={addIntervention} 
          />
      ))*/
    contentRender = contentRenderArray;
    console.log(contentRender);
  }

  return (
    <MainLayout title="Layers">
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

    display: grid;
    justify-items: center;
  }
`;

export default LayersPanel