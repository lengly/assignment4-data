cd cs336_data
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
wget dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin
wget dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_hatespeech_final.bin
cd ..

git config --global user.email "liu_yuda@163.com"
git config --global user.name "lengly"

apt-get update
apt-get install parallel