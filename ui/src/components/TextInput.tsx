import {Input, Tag} from 'antd';
import styled from "styled-components";

interface Props {
    textIntervention: {[key: string]: string}
    setTextIntervention: (newValues: {[key: string]: string}) => void
}

export function TextInput(props: Props): JSX.Element {
    let contentRender = (
        Object.keys(props.textIntervention).map((key) =>
            <WideInput
                placeholder={key}
                value={props.textIntervention[key]}
                onChange={(e) => {props.setTextIntervention({...props.textIntervention, [key]: e.target.value})}}
            ></WideInput>
        )
    )

    return <>{contentRender}</>;
}

const WideInput = styled(Input)`
  width: 100%;
`;