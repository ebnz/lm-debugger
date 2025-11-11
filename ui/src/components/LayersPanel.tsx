import {Empty, Card, Spin, notification} from "antd";
import { LayerPrediction, ValueId } from "../types/dataModel";
import styled from "styled-components";
import {Layer} from "./Layer"
import {useEffect, useMemo} from "react";

interface Props {
  layers?: Array<LayerPrediction>;
  setSelectedValueId: (v: ValueId) => void;
  addIntervention: (valueId: ValueId) => void;
  isLoading: boolean;
  errorMessage?: string;
}

function LayersPanel(props: Props): JSX.Element {
  const {
    layers,
    setSelectedValueId,
    addIntervention,
    isLoading,
    errorMessage,
  } = props;

  useEffect(() => {
    if (errorMessage !== undefined) {
      notification.error({
        message: 'Error',
        description: errorMessage,
        placement: 'topLeft'
      });
    }
  }, [errorMessage]);

  let contentRender: React.ReactNode = <></>;

  contentRender = useMemo(() => layers?.sort((a, b) => a.layer >= b.layer ? 1 : -1)
    .map((item) => (
      <Layer
        key={`layer_${item.type}_${item.layer}`}
        layer={item}
        onAnalyze={valueId => setSelectedValueId(valueId)}
        onCopy={addIntervention}
      />
    )), [layers]);

  if (isLoading) {
    contentRender = <Spin style={{margin: "auto auto"}} tip="Loading prediction" />;
  } else if (layers === undefined){
    contentRender = <Empty description="Run a query to see the predicted layers"/>
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