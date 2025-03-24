from elasticsearch import Elasticsearch


def get_esclient(host, port, api_key):
    return Elasticsearch(
        f"http://{host}:{port}/",
        api_key=api_key
    )


