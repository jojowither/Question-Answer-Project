mkdir -p tmp

curl -X POST http://127.0.0.1:18080/predictions/qa_server \
    -T test_json/sample_text1.txt > tmp/output1&
curl -X POST http://127.0.0.1:18080/predictions/qa_server \
    -T test_json/sample_text2.txt > tmp/output2&
curl -X POST http://127.0.0.1:18080/predictions/qa_server \
    -T test_json/sample_text2.txt > tmp/output3

wait

cat tmp/output1
cat tmp/output2
cat tmp/output3