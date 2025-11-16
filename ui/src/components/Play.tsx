import React, {ChangeEvent, useState} from "react";
import styled from "styled-components";
import { Input, Button, Progress } from 'antd';
import AimOutlined from "@ant-design/icons/AimOutlined";
import ValueLabel from "./ValueLabel";
import {Layer} from "./Layer"
import LayersPanel from "./LayersPanel";
import {LayerPrediction} from "../types/dataModel";

// LayerPrediction
//  layer_name: string;
// predictions_before: Array<Prediction>;
// predictions_after: Array<Prediction>;
// significant_values: Array<ScoredValue>;

function Play(): JSX.Element {
    const labels = [
        {name: "LMDebuggerIntervention", type: "intervention", changeable_layer: false, score: 0.1, layer: 10, dim:2},
        {name: "LMDebuggerIntervention", type: "intervention", changeable_layer: false, score: 0.5, layer: 12, dim:2},
        {name: "LMDebuggerIntervention", type: "intervention", changeable_layer: false, score: 0.85, layer: 4, dim:2},
        {name: "LMDebuggerIntervention", type: "intervention", changeable_layer: false, score: 0.1, layer: 44, dim:2},
        {name: "LMDebuggerIntervention", type: "intervention", changeable_layer: false, score: 0.5, layer: 23, dim:2},
        {name: "LMDebuggerIntervention", type: "intervention", changeable_layer: false, score: 0.85, layer: 200, dim:2},
        {name: "LMDebuggerIntervention", type: "intervention", changeable_layer: false, score: 0.1, layer: 3, dim:2},
        {name: "LMDebuggerIntervention", type: "intervention", changeable_layer: false, score: 0.35, layer: 2, dim:20},
        {name: "LMDebuggerIntervention", type: "intervention", changeable_layer: false, score: 0.71, layer: 202, dim:2},
        {name: "LMDebuggerIntervention", type: "intervention", changeable_layer: false, score: 0.25, layer: 2, dim:10},
    ]
    const input = {
        predictions_before: [
            {"token": "pizza", "score": 0.5},
            {"token": "ice-cream", "score": 0.9},
            {"token": "burger", "score": 0.3},
            {"token": "noodle", "score": 0.5},
            {"token": "hot-dog", "score": 0.9},
            {"token": "kebab", "score": 0.3},
            {"token": "sushi", "score": 0.5},
            {"token": "calzone", "score": 0.9},
        ],
        predictions_after: [
            {"token": "pizza", "score": 0.5},
            {"token": "ice-cream", "score": 0.9},
            {"token": "burger", "score": 0.3},
            {"token": "noodle", "score": 0.5},
            {"token": "hot-dog", "score": 0.9},
            {"token": "kebab", "score": 0.3},
            {"token": "sushi", "score": 0.5},
            {"token": "calzone", "score": 0.9},
        ],
        layer: 10,
        name: "LMDebuggerIntervention",
        type: "intervention",
        changeable_layer: false,
        significant_values: labels,
        text_inputs: {},
        text_outputs: {},
        docstring: "ABC",
        min_layer: 0,
        max_layer: 32
    }
    const n_layers = 5; 
    let layer_inputs = []
    for(let idx = 0; idx < n_layers; idx++) {
        layer_inputs.push(Object.assign({}, input))
        layer_inputs[idx]['layer'] = n_layers - idx;
        layer_inputs[idx]['name'] = input['name'];
        layer_inputs[idx]['type'] = input['type'];
        layer_inputs[idx]['changeable_layer'] = input['changeable_layer'];
    }
    return (
        <LayersPanel 
            layers={layer_inputs}
            addIntervention={(v) => console.log(`L${v.layer}D${v.dim}`)}
            setSelectedValueId={(v) => {console.log(v)}}
            isLoading={false}
            />
    );
}

export default Play;