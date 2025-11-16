import React from "react";
import styled from "styled-components";
import {Button, Tooltip} from 'antd';
import { Prediction } from "../types/dataModel";
import {FontSizeOutlined, NumberOutlined} from '@ant-design/icons';
import { Typography } from 'antd';

const { Text } = Typography;

interface Props {
    predToken: Prediction,
    isFirst: boolean
  }


const TokenButton = styled(Button)`
    padding: 1px 5px;
`;

function TokenLabel(props: Props) : JSX.Element {
    const {
        predToken, 
        isFirst
    } = props;

    const normalText = <Text keyboard >{predToken.token}</Text>
    const firstText = <MyText keyboard >{predToken.token}</MyText>
    const theText = isFirst? firstText : normalText
    return (
            <TokenButton type="ghost">
                {predToken.rank !== undefined ?
                    <><Tooltip title="Rank of the edited Token in Output Distribution"><NumberOutlined />{predToken.rank}</Tooltip> {" "} {theText} </> :
                    <><FontSizeOutlined style={{color: '#686565', fontSize: '10pt'}}/> {theText}</>}
            </TokenButton>
    );
}

const MyText = styled(Text)`
    &.ant-typography kbd {
        background-color: #ace0e4;
    }
`; 

export default TokenLabel;
