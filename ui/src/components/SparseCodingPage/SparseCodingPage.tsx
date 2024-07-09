import React, {useState} from "react";
import { hot } from "react-hot-loader";
import styled, {css} from "styled-components";

import {Button, Col, Divider, Input, Row, notification} from "antd";
import AutoEncodersPanel from "./AutoEncodersPanel";
import {getNeuronActivationPerToken} from "../../api/prediction";
import {AutoEncoderResponse} from "../../types/dataModel";
const {TextArea} = Input;

function SparseCodingPage(): JSX.Element {
    const [api, contextHolder] = notification.useNotification();

    const [text, setText] = useState("");
    const [isLoading, setLoading] = useState(false);
    const [autoencoderResults, setAutoencoderResults] = useState<Array<AutoEncoderResponse>>([])

    async function handleRun(prompt: string, neuron_id: number) {
        if (prompt.length === 0) {
            api.open({
                message: "No Prompt specified!",
                description: "Please enter a prompt into the Text Area."
            });
            return;
        }
        setLoading(true);
        const return_value = await getNeuronActivationPerToken(prompt, neuron_id);
        setAutoencoderResults([return_value]);
        setLoading(false);
    }

    return (<>
        {contextHolder}
        <Row gutter={16}>
            <Col className="gutter-row" span={12}>
                <TextArea rows={10} disabled={isLoading} value={text} onChange={(e) => {setText(e.target.value)}}></TextArea>
            </Col>
            <Col className="gutter-row" span={12}>
                <Button disabled={isLoading} onClick={() => {handleRun(text, 15769)}}>Run</Button>
            </Col>
        </Row>

        <AutoEncodersPanel isLoading={false} errorMessage={undefined} autoencoder_results={autoencoderResults}></AutoEncodersPanel>
    </>)
}

export default SparseCodingPage;