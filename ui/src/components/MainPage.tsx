import React, {useState} from "react";
import { hot } from "react-hot-loader";
import Prompt from "./Prompt";
import {NetworkPrediction, Intervention, ValueId} from "../types/dataModel";
import {predict, generate} from "../api/prediction";
import LayersPanel from "./LayersPanel";
import {MemoValueDetailsPanel} from "./ValueDetailsPanel";
import InterventionsPanel from "./InterventionsPanel";
import styled, {css} from "styled-components";
import {Button, Checkbox, Modal, Space, Tag, Upload} from "antd";
import {DownSquareOutlined} from "@ant-design/icons";
import {useCookies} from "react-cookie";

// Sortable Interventions
import {arrayMove} from "@dnd-kit/sortable";


function MainPage(): JSX.Element {

  const [cookies, setCookie, removeCookie] = useCookies();

  const [promptValue, setPromptValue] = useState<string>("");
  const [prediction, setPrediction] = useState<NetworkPrediction | undefined>(undefined);
  const [interventions, setInterventions] = useState<Array<Intervention>>([]);
  const [selectedValueId, setSelectedValueId] = useState<ValueId | undefined>(undefined);
  const [isLoadingPrediction, setLoadingPrediction] = useState<boolean>(false);
  const [predictionError, setPredictionError] = useState<string | undefined>(undefined);

  const [introOpen, setIntroOpen] = useState<boolean>(cookies.dontShowIntro === undefined || !cookies.dontShowIntro);
  const [dontShowAgain, setDontShowAgain] = useState<boolean>(false);

  // ----------------------------------- //
  // Intervention State Update functions //
  // ----------------------------------- //
  function addIntervention(valueId: ValueId) {
    if(!hasIntervention(valueId)){
      setInterventions(prev => [{...valueId, coeff: 0.0}, ...prev])
    }
  }

  function updateIntervention(valueId: ValueId, coeff: number){
    const {name, layer, dim, desc} = valueId;
    setInterventions(interventions.map(
      (inter) => {
        if (inter.layer === layer && inter.dim === dim && inter.name == name) {
          return {...inter, coeff: coeff};
        }
        return inter;
      }
    ))
  }

  function deleteIntervention (l: number, d: number, t: string){
    setInterventions(interventions.filter(({name, layer, dim}) => (name !== t) || (layer !== l) || (dim !== d)))
  }

  function hasIntervention (valueId: ValueId) {
    return interventions.filter(
        ({layer, dim, name}) =>
            (layer === valueId.layer)
            && (dim === valueId.dim)
            && (name === valueId.name)
    ).length > 0
  }

  function selectIntervention(valueId: ValueId): void {
    setSelectedValueId(valueId)
  }

  function setIndexOfIntervention(oldIdx: number, newIdx: number) {
    if (oldIdx % 1 !== 0 || newIdx % 1 !== 0) {
      return;
    }

    // Rearrange items
    setInterventions((prevItems) => arrayMove(prevItems, oldIdx, newIdx));
  }

  async function handleGenerate(prompt: string, generate_k: Number): Promise<string> {
    setPredictionError(undefined);
    try {
      const result = await generate({prompt, interventions, generate_k});
      const {generate_text} = result;
      return generate_text;
    } catch(e) {
      setPredictionError("Failed generation");
      return prompt;
    }
  }

  async function handleRun(prompt: string){
    setLoadingPrediction(true);
    setPredictionError(undefined);
    try {
      const result = await predict({prompt, interventions, generate_k: 1});
      console.log(result)
      if (result?.layers.length === 0 || result === undefined) {
        setPredictionError("Got empty result. Check your Backend!");
        console.log("Got empty result. Check your Backend!");
      }
      setPrediction(result);
    } catch(e) {
      if (e instanceof Error) {
        setPredictionError(e.message);
      }
      console.log(e)
    } finally {
      setLoadingPrediction(false);
    }
  }

  // ------------------------------- //
  // Serialization of Trace/Generate //
  // ------------------------------- //

  function handleDownload() {
    let result_serialized = JSON.stringify({
      promptValue: promptValue,
      prediction: prediction,
      interventions: interventions
    });
    const blob = new Blob([result_serialized], { type: "application/json" });

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "export.json"; // File name
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleUpload(file: any) {
    const isJson = file.type === 'application/json' || file.name.endsWith('.json');

    if (!isJson) {
      console.error('You can only upload JSON files!');
      return Upload.LIST_IGNORE;
    }

    const reader = new FileReader();

    reader.onload = (e) => {
      try {
        const result = e.target?.result;
        if (typeof result === "string") {
          const parsed = JSON.parse(result);

          // Use it in your app
          setInterventions(parsed.interventions);
          if (parsed.hasOwnProperty("prediction")) {
            setPrediction(parsed.prediction);
          }
          setPromptValue(parsed.promptValue);
          console.log('JSON file uploaded and parsed successfully!');
        } else {
          console.error('JSON-Parsing: invalid file');
        }
      } catch (err) {
        console.error('Error parsing JSON:', err);
      }
    };

    reader.readAsText(file);
  }

  const detailsVisible = selectedValueId !== undefined;

  function handleIntroClose() {
    if (dontShowAgain) {
      setCookie("dontShowIntro", true);
    }
    setIntroOpen(false);
  }

  return (
    <MainLayout detailsVisible={detailsVisible}> 
      <PromptArea>
        <Prompt 
          onRun={handleRun}
          onGenerate={handleGenerate}
          isLoading={isLoadingPrediction}
          promptValue={promptValue}
          setPromptValue={setPromptValue}
        />
      </PromptArea>
      <ValueDetailsArea detailsVisible={detailsVisible}>
        <MemoValueDetailsPanel valueId={selectedValueId} />
      </ValueDetailsArea>
      <LayersViewArea>
        <LayersPanel 
          layers={prediction?.layers}
          setSelectedValueId={setSelectedValueId}
          addIntervention={(valueId) => addIntervention(valueId)}
          isLoading={isLoadingPrediction}
          errorMessage={predictionError}
        />
      </LayersViewArea>
      <InterventionArea>
        <InterventionsPanel 
          interventions={interventions}
          addIntervention={(valueId: ValueId) => addIntervention(valueId)}  
          deleteIntervention={(l, d, t) => deleteIntervention(l, d, t)}
          updateIntervention={(v, c) => updateIntervention(v, c)}
          selectIntervention={(v) => selectIntervention(v)}
          setIndexOfIntervention={setIndexOfIntervention}
          handleDownload={handleDownload}
          handleUpload={handleUpload}
        />
      </InterventionArea>

      <Modal
        visible={introOpen}
        closable={false}
        title="Welcome to LM-Debugger"
        width="900px"
        footer={[
          <Checkbox checked={dontShowAgain} onChange={(e) => {setDontShowAgain(e.target.checked)}}>
            Don't show again
          </Checkbox>,
          <Space size="large"/>,
          <Button key="submit" type="primary" onClick={handleIntroClose}>
            Close
          </Button>
        ]}
      >
        <p>Enter an exemplary Prompt into the textfield</p>
        <p>Use <Tag color="blue">Generate</Tag>generate new Tokens using the Transformer Model</p>
        <p>Use <Tag color="blue">Trace</Tag>to see the Internal Calculations, Intervention Methods and Metrics of the Transformer Model</p>
        <p>Click <CopyIcon /> to intervene on a Feature's Activation or add an Intervention using a Model-Transformation and <Tag color="blue">Add as Intervention</Tag></p>
        <p><Tag color="blue">Generate</Tag>and <Tag color="blue">Trace</Tag>produce different outputs with different Interventions applied</p>
        <p>Model-Transformation Interventions are sortable via drag-and-hold</p>
      </Modal>
    </MainLayout>
  )
}

interface DetailsVisibleProps {
  detailsVisible: boolean;
}

const CopyIcon = styled(DownSquareOutlined)`
    font-size: 15px;
    color: #717D7E;
`;

const withDetails = css`
  grid-template-columns: 5fr 1fr;
`;

const withoutDetails = css`
  grid-template-columns: 1fr 0px;
`;

const MainLayout = styled.div<DetailsVisibleProps>`
  width: 100%;
  height: 100%;

  background-color: #dce0e6;

  padding: 4px;

  display: grid;
  grid-template-columns: 2fr 1fr;
  grid-template-rows: min-content 1fr 140px;

  grid-template-areas: 
    "prompt  details"
    "layers  details"
    "inter   inter";

  ${(props) => props.detailsVisible ? withDetails : withoutDetails}
`;

const PromptArea = styled.div`
  grid-area: prompt;
  padding: 2px;
`;

const ValueDetailsArea = styled.div<DetailsVisibleProps>`
  grid-area: details; 
  visibility: ${(props) => props.detailsVisible ? "visible" : "hidden"};
  padding: ${(props) => props.detailsVisible ? "2px" : "0px"};
`;

const LayersViewArea = styled.div`
  grid-area: layers;  
  padding: 2px;
`;

const InterventionArea = styled.div`
  grid-area: inter;
  padding: 2px;
`;

export default hot(module)(MainPage);
