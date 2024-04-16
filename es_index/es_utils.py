from elasticsearch import Elasticsearch


def get_esclient(host, port, ca_cert_path, api_key):
    return Elasticsearch(
        f"https://{host}:{port}/",
        api_key=api_key,
        verify_certs=True,
        ca_certs=ca_cert_path
    )


