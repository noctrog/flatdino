from dataclasses import asdict
from pathlib import Path
import hashlib
import requests
import json
import os

from osfclient import OSF

from flatdino.data import CACHE_FOLDER, HASH_FOLDER


def download_osf_file(
    project_name: str,
    file_name: str,
    output_path: Path = Path("/tmp"),
    storage_name: str = "osfstorage",
):
    osf = OSF()
    project = osf.project(project_name)
    storage = project.storage(storage_name)

    for file in storage.files:
        if file.name == file_name:
            local_file = output_path / file_name
            with open(local_file, "wb") as f:
                file.write_to(f)
            return local_file

    raise ValueError("file not found in OSF project")


def download_file_from_url(url: str, destination: Path):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)

    return destination


def hash_file(path: Path | str, blocksize: int = 655360):
    path = Path(path) if isinstance(path, str) else path

    assert path.exists()
    assert path.is_file()

    h = hashlib.sha256()
    with open(path.absolute(), "rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dir(directory: Path | str) -> str:
    directory = Path(directory) if isinstance(directory, str) else directory

    assert directory.is_dir()

    h = hashlib.sha256()
    directory = directory.absolute()
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        files.sort()
        for name in files:
            path = os.path.join(root, name)
            relpath = os.path.relpath(path, directory)
            h.update(relpath.encode("utf-8"))  # ty: ignore
            h.update(hash_file(path).encode("utf-8"))
    return h.hexdigest()


def hash_dataclass(dclass) -> str:
    return hashlib.sha256(json.dumps(asdict(dclass), sort_keys=True).encode()).hexdigest()


def get_dataset_root(dataset_name: str) -> Path:
    return Path(f"{CACHE_FOLDER}") / dataset_name


def get_dataset_hash_file(dataset_name: str) -> Path:
    return Path(HASH_FOLDER) / f"{dataset_name}.txt"


def write_hash(dataset_name: str) -> None:
    dataset_root = get_dataset_root(dataset_name)
    dataset_hash = hash_dir(dataset_root)
    HASH_FOLDER.mkdir(parents=True, exist_ok=True)
    get_dataset_hash_file(dataset_name).write_text(dataset_hash)


def is_dataset_healthy(dataset_name: str, force: bool) -> bool:
    if force:
        return False

    dataset_root = CACHE_FOLDER / dataset_name
    if not dataset_root.exists() or not dataset_root.is_dir():
        return False

    hash_file = get_dataset_hash_file(dataset_name)
    if not hash_file.exists():
        return False

    try:
        stored_hash = hash_file.read_text().strip()
        return stored_hash != "" and hash_dir(dataset_root) == stored_hash
    except (OSError, AssertionError):
        return False
