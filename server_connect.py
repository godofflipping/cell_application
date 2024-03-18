from minio import Minio
from minio.error import S3Error

class ServerClient():
    
    def __init__(self):
        path = "client.txt"
        with open(path) as file:
            self.client_data = [line.strip() for line in file]
        
        self.client = Minio(self.client_data[0],
            access_key = self.client_data[1],
            secret_key = self.client_data[2],
            secure = False
        )
        
        self.bucket_name = self.client_data[3]
        

    def upload(self, source_file, destination_file):
        
        found = self.client.bucket_exists(self.bucket_name)
        if not found:
            self.client.make_bucket(self.bucket_name)
            print("Created bucket", self.bucket_name)
        else:
            print("Bucket", self.bucket_name, "already exists")

        self.client.fput_object(
            self.bucket_name, destination_file, source_file,
        )
        print(
            source_file, "successfully uploaded as object",
            destination_file, "to bucket", self.bucket_name,
        )
        
        
    def download(self, download_file, source_file):

        try:
            self.client.fget_object(
                self.bucket_name, source_file, download_file,
            )
            print(
                download_file, "successfully downloaded as object",
                source_file, "to bucket", self.bucket_name,
            )
        except:
            print("Object was not found")
        
        
    def remove(self, object_names):
        
        self.client.remove_objects(
            self.bucket_name, object_names
        )
        print(
            object_names,
            "has been successfully removed from",
            self.bucket_name
        )

if __name__ == "__main__":
    server_client = ServerClient()