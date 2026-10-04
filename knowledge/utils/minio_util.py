from minio import Minio
from minio.error import S3Error
import logging
import os
from dotenv import load_dotenv
load_dotenv()


def get_minio_client():

    #1. 实例化MinIO客户端
    try:
        client = Minio(os.getenv("MINIO_ENDPOINT"),   #服务端端口
                       access_key="minioadmin",
                       secret_key="minioadmin",
                       secure=False,    #使用http协议进行传输,不用https传输
                       )

        #2. 判断桶是否存在,桶要是不存在我们就创建这个桶
        bucket_name = os.getenv("MINIO_BUCKET_NAME")
        bucket_exists = client.bucket_exists(os.getenv("MINIO_BUCKET_NAME"))
        if not bucket_exists:
            client.make_bucket(os.getenv("MINIO_BUCKET_NAME"))
            print("Created bucket", os.getenv("MINIO_BUCKET_NAME"))
            logging.info(f"桶:{bucket_name} 不存在")
        else:
            logging.info(f"桶:{bucket_name} 已经存在")

        return client
    except S3Error as e:
        logging.error("MinIO客户端创建失败")
        return None

if __name__ == "__main__":
    client = get_minio_client()
    print(client)