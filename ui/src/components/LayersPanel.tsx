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

  let metricsContent: React.ReactNode = <></>;
  let interventionContent: React.ReactNode = <></>;

  const renderOverlay = isLoading || layers === undefined;
  let overlayContent: React.ReactNode = <></>;

  metricsContent = useMemo(() => layers?.filter((item) => item.layer === -1 || item.type === "LMDebuggerIntervention")
    .sort((a, b) => a.layer >= b.layer ? 1 : -1)
    .map((item) => (
      <Layer
        key={`layer_${item.type}_${item.layer}`}
        layer={item}
        onAnalyze={valueId => setSelectedValueId(valueId)}
        onCopy={addIntervention}
      />
    )), [layers]);

  interventionContent = useMemo(() => layers?.filter((item) => item.layer !== -1 && item.type !== "LMDebuggerIntervention")
    .sort((a, b) => a.layer >= b.layer ? 1 : -1)
    .map((item) => (
      <Layer
        key={`layer_${item.type}_${item.layer}`}
        layer={item}
        onAnalyze={valueId => setSelectedValueId(valueId)}
        onCopy={addIntervention}
      />
    )), [layers]);

  if (isLoading) {
    overlayContent = <Spin style={{margin: "auto auto"}} tip="Loading prediction" />;
  } else if (layers === undefined){
    overlayContent = <Empty description="Run a query to see the predicted layers"/>;
  }

  return (
    <div style={{display: "flex", flexDirection: "row", height: "80vh", overflow: "auto", position: "relative"}}>
      {renderOverlay && <OverlayLayout>{overlayContent}</OverlayLayout>}
      <MainLayout title="Metrics">{metricsContent}</MainLayout>
      <MainLayout title="Layers">{interventionContent}</MainLayout>
    </div>
  );
}

const OverlayLayout = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 999;
  background-color: rgba(255, 255, 255, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
`;

const MainLayout = styled(Card).attrs({
  size: "small"
})`
  width: 50%;
  height: 100%;

  &.ant-card .ant-card-body {
    height: calc(100vh - 236px);
    overflow-x: hidden;
    overflow-y: auto;
    padding: 2px;

    display: grid;
    justify-items: center;
    align-content: start;
  }
`;

export default LayersPanel