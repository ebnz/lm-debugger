import React, {useState} from "react";
import { hot } from "react-hot-loader";
import styled, {css} from "styled-components";

import {Button, Col, Divider, Input, Row, notification, Select} from "antd";
import AutoEncodersPanel from "./AutoEncodersPanel";
import {get_autoencoder_files, getNeuronActivationPerToken} from "../../api/prediction";
import {AutoEncoderResponse} from "../../types/dataModel";
const {TextArea} = Input;

function SparseCodingPage(): JSX.Element {
    const [api, contextHolder] = notification.useNotification();

    const [text, setText] = useState("");
    const [isLoading, setLoading] = useState(false);

    const [autoencoderIndex, setAutoencoderIndex] = useState<Array<number|null>>([0, null, null]);
    const [autoencoderFeatures, setAutoencoderFeatures] = useState<Array<Array<number>>>([[157, 15769], [], []]);
    const [autoencoderResults, setAutoencoderResults] = useState<Array<Array<AutoEncoderResponse>>>([[], [], []]);

    const [autoencoderSelectValue, setAutoencoderSelectValue] = useState<number>(0);

    let autoencoderSelectOptions: any[] = [];
    get_autoencoder_files().then((names) => {
        for (let index = 0; index < names.length; index++) {
            autoencoderSelectOptions.push({value: index, label: names[index]})
        }
    });



    async function handleRun(prompt: string, ae_output_column: number) {
        if (prompt.length === 0) {
            api.open({
                message: "No Prompt specified!",
                description: "Please enter a prompt into the Text Area."
            });
            return;
        }
        setLoading(true);
        let rv_cache = [];
        for(let neuron_id of autoencoderFeatures[ae_output_column]) {
            const return_value = await getNeuronActivationPerToken(prompt, neuron_id);
            rv_cache.push(return_value);
        }
        let copyAutoEncoderResults = autoencoderResults;
        copyAutoEncoderResults[ae_output_column] = rv_cache;
        setAutoencoderResults(copyAutoEncoderResults);
        setLoading(false);
    }

    function handleFeaturesChange(new_features: Array<number>, ae_column_id: number) {
        let featuresCopy = [...autoencoderFeatures];
        featuresCopy[ae_column_id] = new_features;
        setAutoencoderFeatures(featuresCopy);
    }

    function addNewAutoencoder() {
        //ToDo: Test
        try {
            const index = autoencoderIndex.indexOf(null);
            let autoencoderIndexCopy = [...autoencoderIndex];
            autoencoderIndexCopy[index] = autoencoderSelectValue;
        }
        catch (error) {
            //ToDo: Toast
            return;
        }
    }

    console.log(autoencoderFeatures)


    return (<>
        {contextHolder}
        <Row gutter={16}>
            <Col className="gutter-row" span={12}>
                <TextArea rows={10} disabled={isLoading} value={text} onChange={(e) => {setText(e.target.value)}}></TextArea>
            </Col>
            <Col className="gutter-row" span={12}>
                <Button disabled={isLoading} onClick={() => {handleRun(text, 0)}}>Run</Button>
                <br/>
                <Select
                    style={{width: "120px"}}
                    onChange={(value) => {setAutoencoderSelectValue(parseInt(value))}}
                    options={autoencoderSelectOptions}></Select>
                <Button onClick={() => {addNewAutoencoder()}}></Button>
            </Col>
        </Row>

        <Row gutter={16}>
            <Col className="gutter-row" span={8}>
                <AutoEncodersPanel
                    autoencoderIndex={autoencoderIndex[0]}
                    setAutoencoderIndex={(value: number) => {setAutoencoderIndex([value, autoencoderIndex[1], autoencoderIndex[2]])}} //ToDo: Beautify
                    autoencoderFeatures={autoencoderFeatures[0]}
                    autoencoder_results={autoencoderResults[0]}
                    handleAutoencoderFeaturesChange={(new_features: Array<number>) => {handleFeaturesChange(new_features, 0)}}/>
            </Col>
            <Col className="gutter-row" span={8}>
                <AutoEncodersPanel
                    autoencoderIndex={autoencoderIndex[1]}
                    setAutoencoderIndex={(value: number) => {setAutoencoderIndex([autoencoderIndex[0], value, autoencoderIndex[2]])}} //ToDo: Beautify
                    autoencoderFeatures={autoencoderFeatures[1]}
                    autoencoder_results={autoencoderResults[1]}
                    handleAutoencoderFeaturesChange={(new_features: Array<number>) => {handleFeaturesChange(new_features, 1)}}/>
            </Col>
            <Col className="gutter-row" span={8}>
                <AutoEncodersPanel
                    autoencoderIndex={autoencoderIndex[2]}
                    setAutoencoderIndex={(value: number) => {setAutoencoderIndex([autoencoderIndex[0], autoencoderIndex[1], value])}} //ToDo: Beautify
                    autoencoderFeatures={autoencoderFeatures[2]}
                    autoencoder_results={autoencoderResults[2]}
                    handleAutoencoderFeaturesChange={(new_features: Array<number>) => {handleFeaturesChange(new_features, 2)}}/>
            </Col>
        </Row>
    </>)
}

export default SparseCodingPage;