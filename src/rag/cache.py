from pathlib import Path


def collection_path(persist_directory: str, collection_name: str) -> Path:
    return Path(persist_directory) / collection_name


def database_exists(persist_directory: str, collection_name: str) -> bool:
    path = collection_path(persist_directory, collection_name)
    return path.exists() and any(path.iterdir())


def needs_indexing(persist_directory: str, collection_name: str) -> bool:
    return not database_exists(persist_directory, collection_name)
