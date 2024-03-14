import urllib.request
import os
import sys
import time
import zipfile
import tempfile
import shutil
import glob

class DataDownloader:
    def __init__(self, download_path="data-cache"):
        self.start_time = 0
        self.last_report_time = 0
        self.download_path = download_path
        self.tmp_path = tempfile.TemporaryDirectory() #os.path.join(self.download_path, 'tmp')

    # Derived from https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
    def _reporthook(self, count, block_size, total_size):
        current_time = time.time()
        if count == 0:
            self.start_time = current_time
            self.last_report_time = current_time
            return
        time_since_last_report = current_time - self.last_report_time
        if time_since_last_report > 1:
            self.last_report_time = current_time
            duration = current_time - self.start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))

            if (total_size != -1):
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write("%d%%, %d MB, %d KB/s, %d secs    \r" %
                                 (percent, progress_size / (1024 * 1024), speed, duration))
            else:
                sys.stdout.write("%d MB, %d KB/s, %d secs    \r" %
                                 (progress_size / (1024 * 1024), speed, duration))
            sys.stdout.flush()

    def download(self, url, path):
        """
        Downloads a file from the given URL to the specified path.
        
        If the target file already exists at the destination path, the function will 
        skip the download. Otherwise, it downloads the file to a temporary location 
        and then moves it to the desired path. During the download, a progress indicator 
        is displayed via the reporthook function.
        
        :param url: The URL of the file to be downloaded.
        :param path: The local path where the file should be saved.
        """
        if os.path.exists(path):
            print(f"Skipping download of {path}; it already exists")
        else:
            print(f"Downloading from {url}")
            dest_dir = os.path.dirname(path)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            tmp_path = path + '.tmp'
            urllib.request.urlretrieve(url, tmp_path, self._reporthook)
            os.rename(tmp_path, path)
            print("Download complete")

    def unzip(self, zip_file_path, unzip_path, path_test):
        """
        Extracts the contents of a specified ZIP file to a designated directory. 
        
        If the directory doesn't exist, it will be created. The function will skip 
        the extraction process if a specified sub-directory (given by path_test) 
        already exists within the target extraction directory, suggesting that the 
        unzip operation may have already been performed.
        
        :param zip_file_path: Path to the ZIP file to be extracted.
        :param unzip_path: Path to the directory where the ZIP contents should be extracted.
        :param path_test: Sub-directory name (relative to unzip_path) to check as an indicator 
                          if the unzip operation has previously occurred.
        """
        unpack_dir = os.path.join(unzip_path, path_test)
        if os.path.exists(unpack_dir):
            print(f"Skipping unzip of {zip_file_path} as unzip path exists: {unpack_dir}")
        else:
            print(f"Unzipping {zip_file_path} to {unzip_path}")
            
            # Unzip to a temp directory that we rename at the end
            tmp_unzip_path = f'{unzip_path}_tmp'
            
            os.makedirs(tmp_unzip_path, exist_ok=True)

            absolute_unzip_path = os.path.abspath(tmp_unzip_path)

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    full_path = os.path.join(absolute_unzip_path, member)
                    if len(full_path) > 260:
                        msg = f"Extraction path too long for Windows (Max: 260 chars). It is {len(full_path)} characters. {full_path}"
                        raise ValueError(msg)

                zip_ref.extractall(tmp_unzip_path)
            # Rename the tmp_unzip_path to the final name
            # This way if there is a partial unpacking due to an error there
            # will not be a final directory and the process will be 
            # redone, rather than skipping.
            os.rename(tmp_unzip_path, unzip_path)

    def download_and_unzip(self, url, dataset_name):
        """
        This downloads a zip file from the url and unpacks it into a folder based on the dataset_name
        using the DOWNLOAD_PATH as the base path.
        """
        unzip_path = os.path.join(self.download_path, dataset_name)
        if os.path.exists(unzip_path):
            print(f"Skipping {dataset_name} as unzip path exists: {unzip_path}")
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_file_path = os.path.join(temp_dir, f"{dataset_name}.zip")
                self.download(url, zip_file_path)
                self.unzip(zip_file_path, unzip_path, "")  
    
    def move_files(self, patterns, source_directory, destination_directory):
        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)
            print(f'Making destination directory {destination_directory}')
            
        # Find and move files matching the patterns
        for pattern in patterns:
            for filepath in glob.glob(os.path.join(source_directory, pattern)):
                # Get the filename from the filepath
                filename = os.path.basename(filepath)
                
                # Construct the destination filepath
                destination_filepath = os.path.join(destination_directory, filename)
                
                # Move the file to the destination directory
                shutil.move(filepath, destination_filepath)
                print(f"Moved {filepath} to {destination_filepath}")
    
    # Use this to just keep a subset of the files in the zip file. 
    def download_unzip_keep_subset(self, url, zip_file_patterns, dataset_name):
        unzip_path = os.path.join(self.download_path, dataset_name)
        if os.path.exists(unzip_path):
            print(f"Skipping {dataset_name} as unzip path exists: {unzip_path}")
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
            #temp_dir = self.tmp_path
                zip_file_path = os.path.join(temp_dir, f"{dataset_name}.zip")
                print(f'Downloading to {zip_file_path}')
                self.download(url, zip_file_path)
                extract_path = os.path.join(temp_dir, dataset_name)
                self.unzip(zip_file_path, extract_path, extract_path)
                
                # Only keep a subset of the files to limit the storage used
                self.move_files(zip_file_patterns, extract_path, unzip_path)
        # Outside the block, the temporary directory and its contents will be automatically deleted


