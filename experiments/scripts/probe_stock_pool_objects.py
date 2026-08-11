from __future__ import annotations

import base64
import json
import os

import boto3
import botocore

token = {
    "RGW_TOKEN": {
        "version": 1,
        "type": "ldap",
        "id": os.environ["CEPH_LDAP_ID"],
        "key": os.environ["CEPH_LDAP_KEY"],
    }
}
access_key = base64.b64encode(json.dumps(token).encode()).decode()
client = boto3.client(
    "s3",
    endpoint_url="http://ceph-s3-ssd.prod.highfortfunds.com",
    aws_access_key_id=access_key,
    aws_secret_access_key="",
    config=botocore.client.Config(s3={"addressing_style": "path"}),
)

objects = []
paginator = client.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket="lml.bzw", Prefix="data/"):
    for item in page.get("Contents", []):
        key = item["Key"]
        if "pool" in key.lower():
            objects.append(
                {
                    "key": key,
                    "size": item["Size"],
                    "last_modified": item["LastModified"].isoformat(),
                }
            )
print(json.dumps(objects, ensure_ascii=False, indent=2))
