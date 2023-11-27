import os, zipfile, rarfile, shutil

def print_tree_directory(root_dir, indent=''):
    if not os.path.exists(root_dir):
        print(f"{indent}Invalid directory or file: {root_dir}")
        return

    items = os.listdir(root_dir)

    for index, item in enumerate(sorted(items)):
        item_path = os.path.join(root_dir, item)
        is_last_item = index == len(items) - 1

        if os.path.isfile(item_path) and item_path.endswith('.zip'):
            with zipfile.ZipFile(item_path, 'r') as zip_file:
                print(f"{indent}{'└──' if is_last_item else '├──'} {item} (zip file)")
                zip_contents = zip_file.namelist()
                for zip_item in sorted(zip_contents):
                    print(f"{indent}{'    ' if is_last_item else '│   '}{zip_item}")
        else:
            print(f"{indent}{'└──' if is_last_item else '├──'} {item}")

            if os.path.isdir(item_path):
                new_indent = indent + ('    ' if is_last_item else '│   ')
                print_tree_directory(item_path, new_indent)


def upload_model_list():
    weight_root = "weights"
    models = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            models.append(name)

    index_root = "logs"
    index_paths = []
    for name in os.listdir(index_root):
        if name.endswith(".index"):
            index_paths.append("logs/"+name)

    print(models, index_paths)
    return models, index_paths

def manual_download(url, dst):
    token = os.getenv("YOUR_HF_TOKEN")
    user_header = f"\"Authorization: Bearer {token}\""

    if 'drive.google' in url:
        print("Drive link")
        if 'folders' in url:
            print("folder")
            os.system(f'gdown --folder "{url}" -O {dst} --fuzzy -c')
        else:
            print("single")
            os.system(f'gdown "{url}" -O {dst} --fuzzy -c')
    elif 'huggingface' in url:
        print("HuggingFace link")
        if '/blob/' in url or '/resolve/' in url:
          if '/blob/' in url:
              url = url.replace('/blob/', '/resolve/')
          #parsed_link = '\n{}\n\tout={}'.format(url, unquote(url.split('/')[-1]))
          #os.system(f'echo -e "{parsed_link}" | aria2c --header={user_header} --console-log-level=error --summary-interval=10 -i- -j5 -x16 -s16 -k1M -c -d "{dst}"')
          os.system(f"wget -P {dst} {url}")
        else:
          os.system(f"git clone {url} {dst+'repo/'}")
    elif 'http' in url or 'magnet' in url:
        parsed_link = '"{}"'.format(url)
        os.system(f'aria2c --optimize-concurrent-downloads --console-log-level=error --summary-interval=10 -j5 -x16 -s16 -k1M -c -d {dst} -Z {parsed_link}')


def download_list(text_downloads):
    try:
      urls = [elem.strip() for elem in text_downloads.split(',')]
    except:
      return 'No valid link'

    directories = ['downloads', 'logs', 'weights']
    for directory in directories:
        if not os.path.exists(directory):
            os.mkdir(directory)

    path_download = "downloads/"
    for url in urls:
      manual_download(url, path_download)

    # Tree
    print('####################################')
    print_tree_directory("downloads", indent='')
    print('####################################')

    # Place files
    select_zip_and_rar_files("downloads/")

    models, _ = upload_model_list()
    os.system("rm -rf downloads/repo")

    return f"Downloaded = {models}"


def select_zip_and_rar_files(directory_path="downloads/"):
    #filter
    zip_files = []
    rar_files = []

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".zip"):
            zip_files.append(file_name)
        elif file_name.endswith(".rar"):
            rar_files.append(file_name)

    # extract
    for file_name in zip_files:
        file_path = os.path.join(directory_path, file_name)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(directory_path)

    for file_name in rar_files:
        file_path = os.path.join(directory_path, file_name)
        with rarfile.RarFile(file_path, 'r') as rar_ref:
            rar_ref.extractall(directory_path)

    # set in path
    def move_files_with_extension(src_dir, extension, destination_dir):
        for root, _, files in os.walk(src_dir):
            for file_name in files:
                if file_name.endswith(extension):
                    source_file = os.path.join(root, file_name)
                    destination = os.path.join(destination_dir, file_name)
                    shutil.move(source_file, destination)

    move_files_with_extension(directory_path, ".index", "logs/")
    move_files_with_extension(directory_path, ".pth", "weights/")

    return 'Download complete'


def remove_files(file_list):
    for file in file_list:
        if os.path.exists(file):
            os.remove(file)