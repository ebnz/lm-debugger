import React from "react";
import {AutoEncoderResponse, ValueId} from "../../types/dataModel";
import {Empty, Card, Alert, Spin, Divider, Tag, Tooltip} from "antd";
import styled from "styled-components";
import {SparseFeatureContainer} from "../LabelContainer";

interface TokenDescriptorProps {
    text: string,
    activation: number,
    interpretation: string
}
function TokenDescriptor(props: TokenDescriptorProps): JSX.Element {
    const {text, activation, interpretation} = props;
    const rgba_string = "rgb(255, 0, 0, " + activation / 10 + ")";

    if (text === "<0x0A>") {
        return (
            <>
                <Tooltip title={"Activation: \n" + activation.toFixed(2)}>
                    <text style={{backgroundColor: rgba_string}}>\n</text>
                </Tooltip>
                <br/>
            </>
        )
    }

    return (<>
        <Tooltip title={"Activation: \n" + activation.toFixed(10)}>
            <text style={{backgroundColor: rgba_string}}>{text.replace("▁", " ")}</text>
        </Tooltip>
    </>)
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
    contentRender = <Spin style={{margin: "auto auto"}} tip="Loading Feature Activations" />;
  } else if (errorMessage !== undefined) {
    contentRender = <Alert type="error">{errorMessage}</Alert>
  } else if (autoencoder_results.length === 0){
    contentRender = <Empty description="Enter a prompt and click Run to see Feature Activations."/>
  } else {
    let contentRenderArray = [];

    for (let item of autoencoder_results) {
        for (let index = 0; index < item.tokens_as_string.length; index++) {
            contentRenderArray.push(
                <TokenDescriptor key={index} text={item.tokens_as_string[index]} activation={item.neuron_activations[index]} interpretation={item.interpretations[index]}></TokenDescriptor>
                //<text style={{backgroundColor: rgba_string}}>{item.tokens_as_string[index].replace("▁", " ")}</text>
            );
        }
    }
    contentRender = contentRenderArray;
  }

  return (
      <MainLayout title="Features">
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