from modelscope import snapshot_download
# snapshot_download('BAAI/bge-m3', cache_dir='./models')
snapshot_download('iic/nlp_corom_sentence-embedding_chinese-base-medical', cache_dir='/data/csyData/models')


# import os
# from huggingface_hub import HfApi, hf_hub_download

# # 设置镜像地址
# endpoint = "https://hf-mirror.com"  # 替换为镜像地址

# # 初始化API客户端
# api = HfApi(endpoint=endpoint)

# # 指定仓库ID
# repo_id = "Hu0922/BGE_Medical"

# # 列出仓库中的所有文件
# files = api.list_repo_files(repo_id)
# # print("仓库文件列表:")   # 根据自己的需求选择要下载的文件


# file_list = []
# for file in files:
#     if not file.endswith(".onnx") and not file.endswith(".bin"):
#         print(file)
#         file_list.append(file)

# for file in file_list:
#     downloaded_file_path = hf_hub_download(repo_id=repo_id, local_dir=f"/data/csyData/models/{repo_id}", filename=file, endpoint=endpoint)
#     print(f"文件已下载到: {downloaded_file_path}")