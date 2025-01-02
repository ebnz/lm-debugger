import { Input } from 'antd';

interface Props {
    textIntervention: {[key: string]: string}
    setTextIntervention: (newValues: {[key: string]: string}) => void
}

export function TextInput(props: Props): JSX.Element {
    let contentRender = (
        Object.keys(props.textIntervention).map((key) =>
            <Input
                placeholder={key}
                value={props.textIntervention[key]}
                onChange={(e) => {props.setTextIntervention({...props.textIntervention, [key]: e.target.value})}}
            ></Input>
        )
    )

    return <>{contentRender}</>;
}