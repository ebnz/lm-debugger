import React, {ChangeEvent, useState} from "react";
import { Intervention, ValueId } from "../types/dataModel";
import styled from "styled-components";
import {Card, Button, Input, Upload} from "antd";
import {partial} from "lodash";
import SortableInterventionItem from "./InterventionItem";
import {toType, toAbbr} from "../types/constants";
import {UploadOutlined} from "@ant-design/icons";

// Sortable Interventions
import {
  DndContext,
  useSensors,
  useSensor,
  MouseSensor,
  TouchSensor,
  DragStartEvent,
  DragEndEvent,
  DragOverEvent,
  DragOverlay,
  UniqueIdentifier
} from '@dnd-kit/core';
import { SortableContext, verticalListSortingStrategy, horizontalListSortingStrategy, useSortable, arrayMove } from '@dnd-kit/sortable';
import InterventionItem from "./InterventionItem";


interface Props {
  interventions: Array<Intervention>;
  addIntervention: (valueId: ValueId) => void;
  updateIntervention: (valueId: ValueId, coeff: number) => void;
  deleteIntervention: (layer: number, dim: number, type: string) => void;
  selectIntervention: (valueId: ValueId) => void;
  setIndexOfIntervention: (oldIdx: number, newIdx: number) => void;
  handleDownload: () => void;
  handleUpload: (file: any) => void;
}

function InterventionsPanel(props: Props): JSX.Element {
  const {
    interventions,
    addIntervention,
    updateIntervention,
    deleteIntervention,
    selectIntervention,
    setIndexOfIntervention,
    handleDownload,
    handleUpload
  } = props;

  const [inputContent, setInputContent] = useState<string>("");
  const [activeId, setActiveId] = useState<UniqueIdentifier | null>(null);

  // Set up Draggability-Sensors (Mouse and Touch)
  const sensors = useSensors(
    useSensor(MouseSensor, {
    // Press delay of 250ms, with tolerance of 5px of movement
    activationConstraint: {
      delay: 250,
      tolerance: 5,
    },
  }),
    useSensor(TouchSensor)
  );

  // Handle the start/end of the drag operation (dragging finished)
  const handleDragStart = (event: DragStartEvent) => {
    setActiveId(event.active.id);
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    setActiveId(null);

    if (active.id !== over?.id) {
      // Rearrange items
      const oldIdx = interventions.findIndex((item) => item.type + item.layer + item.dim === active.id);
      const newIdx = interventions.findIndex((item) => item.type + item.layer + item.dim === over?.id);

      // Splice Interventions
      setIndexOfIntervention(oldIdx, newIdx);
    }
  };


  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    setInputContent(e.target.value)
  }

  const isValid = parseInput(inputContent).type === "success";
  const handleAdd = () => {
    const p = parseInput(inputContent);
    if(p.type === "success") {
      addIntervention(p.valueId)
    }
    setInputContent("");
  }

  const activeItem = interventions.find((item) => item.layer === activeId);

  return (
    <MainLayout
      title={
        <TitleLayout>
          <TitleText>Interventions</TitleText>
          <ExportButton onClick={handleDownload}>Export Run</ExportButton>
          <StyledUpload beforeUpload={handleUpload} showUploadList={false}>
            <ImportButton icon={<UploadOutlined />}>Import Run</ImportButton>
          </StyledUpload>
          <ValueInput
            // validInput={isValid || inputContent === ""}
            placeholder="L12D34"
            value={inputContent}
            onChange={handleInputChange}
          />
          <AddButton disabled={!isValid} onClick={handleAdd}>Add</AddButton>
        </TitleLayout>
      }
    >
      <DndContext
        sensors={sensors}
        onDragEnd={handleDragEnd}
      >
        <SortableContext
          items={interventions.map((item) => item.type + item.layer + item.dim)}
          strategy={horizontalListSortingStrategy} // Horizontal sorting
        >
          {
            interventions.map((inter) => (
              <SortableInterventionItem
                intervention={inter}
                deleteIntervention={() => deleteIntervention(inter.layer, inter.dim, inter.type)}
                updateIntervention={partial(updateIntervention, inter)}
                select={partial(selectIntervention, inter)}
              />
            ))
          }
        </SortableContext>

        <DragOverlay dropAnimation={null}>
          {activeItem ? (
            // Render a floating copy of the dragged item
            <InterventionItem
              intervention={activeItem}
              deleteIntervention={() => deleteIntervention(activeItem.layer, activeItem.dim, activeItem.type)}
              updateIntervention={partial(updateIntervention, activeItem)}
              select={partial(selectIntervention, activeItem)}
            />
          ) : null}
        </DragOverlay>

      </DndContext>
    </MainLayout>
  );
}

interface ParseSuccess {
  type: "success";
  valueId: ValueId;
}

interface ParseFailed {
  type: "failed";
  msg: string;
}

type ParseResult = ParseSuccess | ParseFailed;

const parseInput = (str: string): ParseResult => {
  const pattern = /^([a-zA-Z])(\d+)D(\d+)$/i
  const arr = pattern.exec(str);
  if (arr !== null) {
    const type_abbr = arr[1];
    if (!toType.has(type_abbr)) {
      return {
      type: "failed",
      msg: "Intervention Method Abbreviation " + type_abbr + " does not exist or is not registered!"
    }
    }
    const type = toType.get(arr[1]);
    const layer = parseInt(arr[2]);
    const dim = parseInt(arr[3]);
    return {
      type: "success",
      valueId: {type: type ?? "", layer: layer, dim: dim}
    }
  } else {
    return {
      type: "failed",
      msg: "Input must be e.g. 'L12D43' or 'S3D69'"
    }
  }
}

const MainLayout = styled(Card).attrs({
  size: "small"
})`
  width: 100%;
  height: 100%;

  &.ant-card .ant-card-body {
    height: 100px;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 2px;

    display:grid;
    grid-auto-flow: column;
    grid-auto-columns: min-content;
    gap: 4px;
  }
`;

const TitleLayout = styled.div`
  display: grid;
  grid-template-columns: min-content 10px min-content 10px min-content 1fr 150px min-content;
  grid-template-rows: 1fr;
  gap: 4px;
  grid-template-areas: 
    "text . button_export . upload . input button_add";

  align-items: center;
`;

const TitleText = styled.span`
  grid-area: text;
`;

const ValueInput = styled(Input)`
  grid-area: input;
`;

// interface VProps {
//   validInput: boolean;
// };
//
// const ValueInput = styled<VProps>(Input)`
//   grid-area: input;
//   border: 0.5px solid ${(props) => props.validInput ? "#40a9ff": "#d73027"};
//   &:hover {
//     border-color: ${(props) => props.validInput ? "#40a9ff": "#d73027"};
//   };

//   &:focus {
//     border-color: ${(props) => props.validInput ? "#40a9ff": "#d73027"};
//   };
// `;

const ExportButton = styled(Button).attrs({
  type: "primary",
  size: "small"
})`
  grid-area: button_export;
  height: 32px;
`;

const ImportButton = styled(Button).attrs({
  type: "primary",
  size: "small"
})`
  grid-area: button_import;
  height: 32px;
`;

const StyledUpload = styled(Upload).attrs({
  size: "small"
})`
  grid-area: upload;
  height: 32px;
`;

const AddButton = styled(Button).attrs({
  type: "primary",
  size: "small"
})`
  grid-area: button_add;
  height: 32px;
`;

export default InterventionsPanel;