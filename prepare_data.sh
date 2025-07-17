cd data
shuf -n 100000 enwiki-20240420-extracted_urls.txt > subsampled_positive_urls_100k.txt
split -n l/16 subsampled_positive_urls_100k.txt pos_urls_part_
# ls pos_urls_part_* | parallel --bar -j 16 "wget --tries=1 --timeout=2 --max-redirect=5 --quota=2m --user-agent='Mozilla/5.0' --read-timeout=2 --dns-timeout=2 --connect-timeout=2 -i {} --warc-file=subsampled_pos_{#} -O /dev/null -o log_{#}"
# cat pos_urls_part_* | parallel --bar -j 16 "timeout 10 wget --tries=1 --timeout=2 --max-redirect=5 --quota=2m --user-agent='Mozilla/5.0' --read-timeout=2 --dns-timeout=2 --connect-timeout=2 {} --warc-file=subsampled_pos_{%} -O /dev/null -a log_{%}"

cat pos_urls_part_* | parallel --bar -j 16 "timeout 10 wget --tries=1 --user-agent='Mozilla/5.0' --read-timeout=2 {} --warc-file=subsampled_pos_{#} -O /dev/null -a log_{%}"

cd ..