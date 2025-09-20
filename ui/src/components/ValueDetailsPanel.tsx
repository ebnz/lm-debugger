import React, {useEffect, useState} from "react";
import styled from "styled-components";
import {ValueId, ValueInterpretation} from "../types/dataModel";
import {Empty, Spin, Table, Card, Alert, Tag} from "antd";
import {getValueInterpretation} from "../api/prediction"
import {v4 as UUIDv4} from "uuid";
import { Typography, Input, Button, Space, Divider } from 'antd';
import runConfig from "../runConfig.json";
const { Text, Title } = Typography;

interface Props {
  valueId?: ValueId;
}


function fixToken(token: string): string {
  return token.replace("Ġ", " ")
}


function ValueDetailsPanel(props: Props): JSX.Element {
  const {
    valueId
  } = props;

  const [isLoading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | undefined>(undefined);
  const [interpretation, setInterpretation] = useState<ValueInterpretation | undefined>(undefined);

  useEffect(() => {
    if (valueId !== undefined) {
      setLoading(true);
      setError(undefined);
      setInterpretation(undefined);
      const [resultPromise, abort] = getValueInterpretation(valueId);
      
      resultPromise.then((r) => {
        setInterpretation(r)
        setLoading(false)
      }).catch((e) => {
        setError("Failed getting details")
        setLoading(false);
      });

      return abort;
    }
    return () => {};
  }, [valueId]);




  let renderedContent: React.ReactNode = <></>;
  if (valueId === undefined) {
    renderedContent = (<Empty description="Select a value to see details" />);
  } else if (isLoading) {
    renderedContent = (<Spin tip="Loading..." />);
  } else if (error !== undefined) {
    renderedContent = (<Alert message={error} type="error" />);
  } else if (interpretation !== undefined && interpretation !== null) {
    console.log(interpretation)
    const dataSource = interpretation.top_k.map(v => ({token: fixToken(v.token), logit: v.logit.toFixed(3),  key: UUIDv4()}));
    renderedContent = (
      <ValuesTable 
        pagination={false}
        dataSource={dataSource}
        columns={COLUMNS_DEFINITION} 
      />)
  }

  if (valueId === undefined) {
    return (
      <MainLayout title="Value Vector Details">
        {renderedContent}
      </MainLayout>
    )
  }

  const title = (
    <div>
      <Space size="small" split={<Divider type="vertical" />}>
        <Text>Value Vector Details</Text>
      </Space>
      <br/><br/>
      <Space size="small" split={<Divider type="vertical" />}>
        <div>
        <Text> Layer <Tag color="#a55397">{valueId.layer}</Tag></Text>
        <Text>Dim. <Tag color="#078e9e">{valueId.dim}</Tag></Text>
        </div>
        <Text strong>{valueId.desc}</Text>
      </Space>
    </div>
  );

  return (
    <MainLayout title={title}>
      {/* <Meta description={valueId?.desc} /> */}
      {renderedContent}
    </MainLayout>
  );
}

const COLUMNS_DEFINITION = [
  {
    title: 'Token',
    dataIndex: 'token',
    key: 'token',
    render: (token: string) => (
      <Text keyboard>{token}</Text>
    )
  },
  {
    title: 'Logit',
    dataIndex: 'logit',
    key: 'logit',
  },
]

const MainLayout = styled(Card).attrs({
  size: "small"
})`
  width: 100%;
  height: 100%;

  &.ant-card .ant-card-body {
    height: calc(100% - 39px);
    overflow-y: auto;

    display: grid;
    justify-items: center;
  }
`;


const ValuesTable = styled(Table)`
  width: 100%;
  height: 100%;

  & thead.ant-table-thead th.ant-table-cell {
    font-weight: bold;
  }
`;


export default ValueDetailsPanel;
