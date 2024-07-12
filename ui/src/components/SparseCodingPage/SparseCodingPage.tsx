import React, {useEffect, useState} from "react";
import { hot } from "react-hot-loader";
import styled, {css} from "styled-components";

import {Button, Col, Divider, Input, Row, notification, Select} from "antd";
import AutoEncodersPanel from "./AutoEncodersPanel";
import {activate_autoencoder, get_autoencoder_files, getNeuronActivationPerToken} from "../../api/prediction";
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

    const [autoencoderSelectOptions, setAutoencoderSelectOptions] = useState<any[]>([]);
    useEffect(() => {
        get_autoencoder_files().then((names) => {
            let autoencoderSelectOptionsCopy = [];
            for (let index = 0; index < names.length; index++) {
                autoencoderSelectOptionsCopy.push({value: index, label: names[index]})
            }
            setAutoencoderSelectOptions(autoencoderSelectOptionsCopy);
        });
    }, []);



    //ToDo: Beautify
    async function handleRun(prompt: string) {
        if (prompt.length === 0) {
            api.open({
                message: "No Prompt specified!",
                description: "Please enter a prompt into the Text Area."
            });
            return;
        }
        setLoading(true);
        let copyAutoEncoderResults: Array<Array<AutoEncoderResponse>> = autoencoderResults;
        for (let ae_output_column = 0; ae_output_column < 3; ae_output_column++) {
            if (autoencoderIndex[ae_output_column] === null) {
                continue;
            }
            //Activate AutoEncoder first
            await activate_autoencoder(autoencoderIndex[ae_output_column]).then((rv) => {
                if (!rv) {
                    api.open({
                        message: "Error activating AutoEncoder!",
                        description: "An Error occurred in the process of activating an AutoEncoder. It seems that this AutoEncoder does not exist."
                    });
                }
            });

            let rv_cache = [];
            for(let neuron_id of autoencoderFeatures[ae_output_column]) {
                const return_value = await getNeuronActivationPerToken(prompt, neuron_id);
                rv_cache.push(return_value);
            }
        copyAutoEncoderResults[ae_output_column] = rv_cache;
        }
        setAutoencoderResults(copyAutoEncoderResults);
        setLoading(false);
    }

    function handleFeaturesChange(new_features: Array<number>, ae_column_id: number) {
        let featuresCopy = [...autoencoderFeatures];
        featuresCopy[ae_column_id] = new_features;
        setAutoencoderFeatures(featuresCopy);
    }

    function addNewAutoencoder() {
        const index = autoencoderIndex.indexOf(null);

        if (index === -1) {
            api.open({
                message: "No free Column!",
                description: "Please free up a column for an AutoEncoder to be placed into."
            });
            return;
        }

        let autoencoderIndexCopy = [...autoencoderIndex];
        autoencoderIndexCopy[index] = autoencoderSelectValue;
        setAutoencoderIndex(autoencoderIndexCopy);
    }


    return (<>
        {contextHolder}
        <Row gutter={16}>
            <Col className="gutter-row" span={12}>
                <TextArea rows={10} disabled={isLoading} value={text} onChange={(e) => {setText(e.target.value)}}></TextArea>
            </Col>
            <Col className="gutter-row" span={12}>
                <Button disabled={isLoading} onClick={() => {handleRun(text)}}>Run</Button>
                <br/>
                <Select
                    style={{width: "120px"}}
                    value={autoencoderSelectValue}
                    onChange={(value) => {setAutoencoderSelectValue(value)}}
                    options={autoencoderSelectOptions}></Select>
                <Button onClick={() => {addNewAutoencoder()}}>Add</Button>
            </Col>
        </Row>

        <Row gutter={16}>
            <Col className="gutter-row" span={8}>
                <AutoEncodersPanel
                    autoencoderIndex={autoencoderIndex[0]}
                    setAutoencoderIndex={(value: number) => {setAutoencoderIndex([value, autoencoderIndex[1], autoencoderIndex[2]])}} //ToDo: Beautify
                    autoencoderFeatures={autoencoderFeatures[0]}
                    autoencoder_results={autoencoderResults[0]}
                    handleAutoencoderFeaturesChange={(new_features: Array<number>) => {handleFeaturesChange(new_features, 0)}}
                    api={api}/>
            </Col>
            <Col className="gutter-row" span={8}>
                <AutoEncodersPanel
                    autoencoderIndex={autoencoderIndex[1]}
                    setAutoencoderIndex={(value: number) => {setAutoencoderIndex([autoencoderIndex[0], value, autoencoderIndex[2]])}} //ToDo: Beautify
                    autoencoderFeatures={autoencoderFeatures[1]}
                    autoencoder_results={autoencoderResults[1]}
                    handleAutoencoderFeaturesChange={(new_features: Array<number>) => {handleFeaturesChange(new_features, 1)}}
                    api={api}/>
            </Col>
            <Col className="gutter-row" span={8}>
                <AutoEncodersPanel
                    autoencoderIndex={autoencoderIndex[2]}
                    setAutoencoderIndex={(value: number) => {setAutoencoderIndex([autoencoderIndex[0], autoencoderIndex[1], value])}} //ToDo: Beautify
                    autoencoderFeatures={autoencoderFeatures[2]}
                    autoencoder_results={autoencoderResults[2]}
                    handleAutoencoderFeaturesChange={(new_features: Array<number>) => {handleFeaturesChange(new_features, 2)}}
                    api={api}/>
            </Col>
        </Row>
    </>)
}

export default SparseCodingPage;