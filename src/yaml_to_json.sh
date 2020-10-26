# Install yq:
# $ pip3 install yq

find . -name "*.yaml" | while read file
do
  filename=$(basename -- "$file")
#  mkdir -p "json/$(dirname "$file")"
  json_name="$(dirname "$file")/${filename%.*}.json"
  yq . "$file" > "$json_name"
  echo "$file --> $json_name"
done