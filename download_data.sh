URL="https://zenodo.org/api/records/14680012/files-archive"
OUTPUT_DIR="./data/subtask-2"
OUTPUT_FILE="$OUTPUT_DIR/full.zip"

mkdir -p "$OUTPUT_DIR"
curl -L "$URL" -o "$OUTPUT_FILE"
unzip "$OUTPUT_FILE" -d "$OUTPUT_DIR"
rm "$OUTPUT_FILE"


URL="https://zenodo.org/api/records/14699130/files-archive"
OUTPUT_DIR="./data/subtask-1"
OUTPUT_FILE="$OUTPUT_DIR/full.zip"

mkdir -p "$OUTPUT_DIR"
curl -L "$URL" -o "$OUTPUT_FILE"
unzip "$OUTPUT_FILE" -d "$OUTPUT_DIR"
rm "$OUTPUT_FILE"

gunzip ./data/subtask-1/*.gz

URL="https://msmarco.z22.web.core.windows.net/msmarco/dev_v2.1.json.gz"
OUTPUT_DIR="./data/marco_v2.1_qa_dev"
OUTPUT_FILE="$OUTPUT_DIR/dev_v2.1.json.gz"
mkdir -p "$OUTPUT_DIR"
curl -L "$URL" -o "$OUTPUT_FILE"
gunzip $OUTPUT_DIR/*.gz



URL="https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2.1_doc_segmented.tar"
OUTPUT_DIR="/data/group_data/cx_group/temporary/marco_v2.1_segmented"
mkdir -p "$OUTPUT_DIR"
curl -L "$URL" -o "$OUTPUT_DIR/msmarco_v2.1_doc_segmented.tar"

cd "$OUTPUT_DIR"
tar -xf "msmarco_v2.1_doc_segmented.tar"
find "$OUTPUT_DIR" -name "*.gz" -exec gzip -d {} \;
