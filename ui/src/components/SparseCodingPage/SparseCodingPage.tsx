import React, {useState} from "react";
import { hot } from "react-hot-loader";
import styled, {css} from "styled-components";

import {Input} from "antd";
import AutoEncodersPanel from "./AutoEncodersPanel";
const {TextArea} = Input;

function SparseCodingPage(): JSX.Element {
    return (<>
        <TextArea rows={10}></TextArea>
        <AutoEncodersPanel isLoading={false} errorMessage={undefined} autoencoder_results={[]}></AutoEncodersPanel>
    </>)
}

export default SparseCodingPage;