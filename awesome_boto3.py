import boto3
from io import StringIO
from src import parameters

client = boto3.resource('s3', **parameters.aws_config)


def list_all_buckets():
    for bucket in client.buckets.all():
        print(bucket)


def connect_to_a_specific_bucket(bucket_name):
    my_bucket = client.Bucket(bucket_name)


def list_folders_in_a_bucket(bucket_name):
    my_bucket = client.Bucket(bucket_name)
    for folder in my_bucket.objects.all():
        print(folder.key)


def get_files_in_a_directory(bucket_name, prefix):    # Prefix structure like 'input/cv/2019-11-11_1/'
    list_of_objects = []
    s3_bucket = client.Bucket(bucket_name)
    list_of_files_in_the_prefix = s3_bucket.objects.filter(Prefix=prefix)
    for file in list_of_files_in_the_prefix:
        print(file.key)
        object = file.key.split('/')[-1]
        list_of_objects.append(object)
    return list_of_objects


def read_objects_from_a_directory(bucket_name, directory, list_of_objects):
    for object in list_of_objects:
        s3_object = client.Object(bucket_name=bucket_name, key=f'{directory}{object}')
        content = StringIO(s3_object.get()['Body'].read().decode('utf-8'))
        print(object, content.read())


def create_new_folder(bukcet_name, folder_path):    # folder path will be like input/cv/new_folder_name/
    try:
        client.put_object(Bucket=bukcet_name, Key=folder_path)
        print("folder created !!")
    except Exception as e:
        print(e)


def delete_a_folder(bucket_name, folder_path):
    client.Bucket(bucket_name).objects.filter(Prefix=folder_path).delete()
    print("folder deleted!!")


def delete_an_object(bucket_name, key_prefix):
    client.Object(bucket_name, key_prefix).delete()


def copy_object(source_bucket_name, source_object_name,
                destination_bucket_name, destination_object_name):
    copy_source = {'Bucket': source_bucket_name, 'Key': source_object_name}

    s3 = boto3.client('s3', **parameters.aws_config)

    s3.copy_object(CopySource=copy_source, Bucket=destination_bucket_name,
                   Key=destination_object_name)


# bucket_name = 'cv-sorting'
# other_bucket = 'batch_upload_files_testing'
# function calls
# object_list = get_files_in_a_directory('cv-sorting', 'input/cv/2019-11-11_1/')
# # print(len(object_list))
# print(object_list)
# list_all_buckets()
# list_folders_in_a_bucket(bucket_name)
# read_objects_from_a_directory(bucket_name, 'input/cv/2019-11-11_1/', object_list)
# delete_a_folder(bucket_name, f'input/cv/{other_bucket}')

# for objects in client.list_objects(Bucket=bucket_name, Prefix='input/cv/2019-11-11_1/')['Contents']:
#     files = objects['Key']
#     print("file path ", files)
#     file = objects['Key'].split('/')[-1]
#     file_path = f'input/cv/2019-11-11_1/{file}'
#     copy_source = {"Bucket": bucket_name, "Key": files}
#     # client.meta.client.copy(copy_source, other_bucket, f'input/cv/destined_bucket/{file}')
#     print(f"moved {file}")

# mark = []
# try:
#     assert len(mark) != 0, "error"
# except AssertionError:
#     print("no ok")


