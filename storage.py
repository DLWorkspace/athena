
## To install azure storage SDK package:
## sudo pip3 install azure azure-storage-queue azure-storage-common azure-storage-blob azure-cosmosdb-table


import time
import uuid


import config as config

from azure.storage.queue import QueueService
from azure.storage.blob import BlockBlobService

from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity

def queue():

    account_name = config.STORAGE_ACCOUNT_NAME
    account_key = config.STORAGE_ACCOUNT_KEY

    queue_service = QueueService(account_name=account_name, account_key=account_key)


    print("Creating task queue")
    task_queue_name = config.TASK_QUEUE_NAME
    queue_service.create_queue(task_queue_name)
    print("Task queue created")
    

    queue_service.put_message(task_queue_name, u'message1')


    messages = queue_service.get_messages(task_queue_name, num_messages=16)
    for message in messages:
        print(message.content)
        queue_service.delete_message(task_queue_name, message.id, message.pop_receipt)        


def blob():
    account_name = config.STORAGE_ACCOUNT_NAME
    account_key = config.STORAGE_ACCOUNT_KEY
    block_blob_service = BlockBlobService(account_name=account_name, account_key=account_key)

    container_name = config.BLOB_CONTAINER_NAME

    block_blob_service.create_container(container_name) 


    img = load_img("/share1/public/isic/images/2.NV/ISIC_0034320.jpg",target_size=(224,224))
    img = img_to_array(img)
    imgbytes = img.tobytes()


    block_blob_service.create_blob_from_bytes(container_name, "test1",imgbytes)

    ib = block_blob_service.get_blob_to_bytes(container_name,"test1").content  

    print(imgbytes == ib)

    print("\nList blobs in the container")
    generator = block_blob_service.list_blobs(container_name)
    for blob in generator:
        print("\t Blob name: " + blob.name)


def table():
    account_name = config.STORAGE_ACCOUNT_NAME
    account_key = config.STORAGE_ACCOUNT_KEY
    table_service = TableService(account_name=account_name, account_key=account_key)
    table_name = config.TABLE_NAME
    #table_service.create_table(table_name)


    imageId = str(uuid.uuid4())
    task = Entity()
    task.PartitionKey = 'dlws'
    task.RowKey = imageId
    task.description = 'test'
    table_service.insert_or_replace_entity(table_name, task)


    task = table_service.get_entity(table_name, 'dlws', imageId)
    print(task.description)

    tasks = table_service.query_entities('tasktable')
    for task in tasks:
        print(task.description)
        print(task.RowKey)



class Azure_Storage():
    def __init__(self,create_new = False):
        account_name = config.STORAGE_ACCOUNT_NAME
        account_key = config.STORAGE_ACCOUNT_KEY        


        self.task_queue_name = config.TASK_QUEUE_NAME
        self.table_name = config.TABLE_NAME
        self.container_name = config.BLOB_CONTAINER_NAME
        self.ImagePartitionKey = config.IMAGE_PARTITION_KEY

        self.table_service = TableService(account_name=account_name, account_key=account_key)
        self.block_blob_service = BlockBlobService(account_name=account_name, account_key=account_key)
        self.queue_service = QueueService(account_name=account_name, account_key=account_key)
        
        if create_new:
            queue_service.create_queue(task_queue_name)
            block_blob_service.create_container(container_name) 
            table_service.create_table(table_name)

    def put_image(self,image_uuid, image_bytes):
        ret = self.block_blob_service.create_blob_from_bytes(self.container_name, image_uuid,image_bytes)
        return ret

    def get_image(self,image_uuid):
        ret = self.block_blob_service.get_blob_to_bytes(self.container_name,image_uuid).content  
        return ret


    def put_classification_result(self,image_uuid,results):
        task = Entity()
        task.PartitionKey = self.ImagePartitionKey
        task.RowKey = image_uuid
        task.results = str(results)
        ret = self.table_service.insert_or_replace_entity(self.table_name, task)        
        return ret

    def get_classification_result(self,image_uuid):
        try:
            task = self.table_service.get_entity(self.table_name, self.ImagePartitionKey, image_uuid)
            return task.results
        except Exception as e:
            return None


    def put_task(self,taskmsg):
        ret = self.queue_service.put_message(self.task_queue_name, taskmsg)
        return ret

    #payload is in message.content
    def get_task(self,num_messages=16):
        messages = self.queue_service.get_messages(self.task_queue_name, num_messages=num_messages, visibility_timeout=1*60)
        return messages
           

    def delete_task(self,message):
        ret = self.queue_service.delete_message(self.task_queue_name, message.id, message.pop_receipt)
        return ret

if __name__ == '__main__':
    table()