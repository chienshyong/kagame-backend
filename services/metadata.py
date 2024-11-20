# Contains code to manage database "metadata", like the buckets in catalogue

from services.mongodb import metadata_collection, catalogue, CATALOGUE_COLLECTION_NAME
from bson import ObjectId
from typing import TypedDict
import random
from tqdm import tqdm


class CatalogueMetadata(TypedDict):  # For catalogue
    _id: ObjectId
    collection_name: str = CATALOGUE_COLLECTION_NAME
    bucket_count: int = 1


def get_catalogue_metadata() -> CatalogueMetadata:
    result = metadata_collection.find_one_and_update(filter={"collection_name": CATALOGUE_COLLECTION_NAME},
                                                     update={"$setOnInsert": CatalogueMetadata()},
                                                     upsert=True,
                                                     return_document=True)
    return result


def update_catalogue_metadata(metadata: CatalogueMetadata) -> None:
    return metadata_collection.find_one_and_update(filter={"collection_name": CATALOGUE_COLLECTION_NAME},
                                            update={"$set": metadata},
                                            return_document=True)


# Put items into random buckets. Try not to do it too much as it's slow.
def bucketize_catalogue_items(bucket_count: int):
    if (bucket_count <= 0):
        raise Exception(f"bucket_count must be at least 0, bucket_count={bucket_count}")
    for item in tqdm(catalogue.find()):
        bucket_num = random.randint(1, bucket_count)
        catalogue.update_one({"_id": item["_id"]}, {"$set": {"bucket_num": bucket_num}})
    metadata = get_catalogue_metadata()
    metadata["bucket_count"] = bucket_count
    update_catalogue_metadata(metadata)