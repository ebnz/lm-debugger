import React from "react";
import styled from "styled-components";
import TokenLabel from "./TokenLabel";
import {ScoredValue, Prediction, ValueId, AutoEncoderResponse} from "../types/dataModel";
import {ValueLabelWithCopy, AutoEncoderFeatureLabel} from "./ValueLabelWithCopy";



interface Props {
    valueLabels: Array<ScoredValue>;
    onAnaylze: (valueId: ValueId) => void;
    onCopy: (valueId: ValueId) => void;
}

export function LabelContainer(props: Props): JSX.Element {
    const labels = props.valueLabels.map(label => 
        <ValueLabelWithCopy
            scoredValue={label}
            key={`L${label.layer}D${label.dim}`} 
            onAnalyze={props.onAnaylze}
            onCopy={props.onCopy} 
        />
    );
    return (
        <ContainerLayout>
            {labels}
        </ContainerLayout>
    );
}

interface PredProps {
    predictions: Array<Prediction>;
}

export function PredictionContainer(props: PredProps) : JSX.Element {
    const predictions = props.predictions.map((pred, idx) => (
       <TokenLabel key={idx.toString() + pred.token} predToken={pred} isFirst={idx === 0}/>
    ));

    return (
        <ContainerLayout>
            {predictions}
        </ContainerLayout>
    );
}

//Sparse Coding

interface SFCProps {
    autoencoder_result: AutoEncoderResponse
}

export function SparseFeatureContainer(props: SFCProps): JSX.Element {
    let {autoencoder_result} = props;

    let labels = [];

    for (let i = 0; i < autoencoder_result.tokens_as_string.length; i++) {
        labels.push(
            <AutoEncoderFeatureLabel
            name={autoencoder_result.tokens_as_string[i]}
            key={i}
            score={autoencoder_result.neuron_activations[i]}
            onAnalyze={() => {}}    //ToDo
            onCopy={() => {}}       //ToDo
        />
        )
    }

    return (
        <ContainerLayout>
            {labels}
        </ContainerLayout>
    );
}

const ContainerLayout = styled.div`
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    justify-content: flex-start;
    align-items: flex-start;
    gap: 4px;
    padding-left: 1em;
    
`;
