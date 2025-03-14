#!/bin/bash

# 设置下载链接和目标文件夹
URL="$1"
TEMP="./temp_random"

# 检查目标目录是否传递
if [ -z "$DEST_DIR" ]; then
  echo "请提供链接！"
  exit 1
fi

# 下载文件
echo "正在下载文件..."
wget -O downloaded_file "$URL"

# 获取文件的扩展名
EXT="${URL##*.}"

# 根据文件类型解压文件
echo "正在解压文件..."

if [ "$EXT" == "zip" ]; then
  # 如果是 zip 文件，使用 unzip 解压
  unzip downloaded_file -d $TEMP
elif [ "$EXT" == "tar" ] || [ "$EXT" == "gz" ]; then
  # 如果是 tar 或 tar.gz 文件，使用 tar 解压
  tar -xvf downloaded_file -C $TEMP
else
  echo "不支持的文件格式：$EXT"
  exit 1
fi

# 转移到指定的位置
# 检查目标目录是否存在，如果不存在则创建
move_to_dir(){
    DEST_DIR="$1"

    if [ ! -d "$DEST_DIR" ]; then
    echo "目录不存在，正在创建 $DEST_DIR ..."
    mkdir -p "$DEST_DIR"
    fi
    mv "$TEMP/$DEST_DIR" $DEST_DIR
}

move_to_dir "DexiNed/BIPEDv2"
move_to_dir "DexiNed/checkpoints"
move_to_dir "PiDiNet/pidinet-master/trained_models"
move_to_dir "PiDiNet/pidinet-master/training_logs"


# 清理下载的压缩文件
echo "清理临时文件..."
rm -r $TEMP

echo "完成"
