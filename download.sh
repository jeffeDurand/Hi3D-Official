mkdir -p ./train/target
rm -rf ./train/rejectedPath/*
gsutil -m cp -r -n gs://train-stor/modelsSpiral32/* ./train/target
python del_targets.py