import React from "react";
import {AutoEncoderResponse, ValueId} from "../../types/dataModel";
import {Empty, Card, Alert, Spin, Divider, Tag } from "antd";
import styled from "styled-components";
import {SparseFeatureContainer} from "../LabelContainer";

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
        <MyDivider orientation="left" orientationMargin="15px">Requested Features: </MyDivider>
      </LayerLayout>
    )
}


interface AEPanelProps {
  isLoading: boolean,
  errorMessage?: string,
  autoencoder_results: Array<AutoEncoderResponse>;
}

function AutoEncodersPanel(props: AEPanelProps): JSX.Element {
  const {
    isLoading,
    errorMessage,
    autoencoder_results,
  } = props;

  let contentRender: React.ReactNode = [];
  if (isLoading) {
    contentRender = <Spin style={{margin: "auto auto"}} tip="Loading prediction" />;
  } else if (errorMessage !== undefined) {
    contentRender = <Alert type="error">{errorMessage}</Alert>
  } else if (autoencoder_results.length === 0){
    contentRender = <Empty description="Run a query to see the predicted layers"/>
  } else {
    let contentRenderArray = [];

    for (let item of autoencoder_results) {
        contentRenderArray.push(
            <AutoEncoderLayer response={item} onAnalyze={() => {}} onCopy={() => {}}/>
        )
    }
    contentRender = contentRenderArray;
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