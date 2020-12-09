check_pwd() {
    if [ ! -d "handlers" ]; then
        echo 'handlers not found'
        exit 1
    fi
}

pack_and_enable_model() {
    handler_file=$1
    model_name=$2
    export_path=$3
    others=$4

    echo ''
    echo '=========model name========='
    echo $model_name

    torch-model-archiver \
        --model-name $model_name \
        --version 1.0 \
        --serialized-file dummy_file \
        --export-path $export_path \
        --handler $handler_file \
        --force \
        --runtime python3
}

main() {
    check_pwd

    torchserve --stop
    sleep 1.5

    model_name=qa_server
    handler_file=handlers/dummy_handler.py
    export_path=engine_serve_mars
    register_model_name=qa_server_batch
    others="batch_size=2&max_batch_delay=100&initial_workers=1&model_name=$register_model_name"
    mkdir -p $export_path

    pack_and_enable_model $handler_file $model_name $export_path $others
    sleep 1.5

    torchserve --start --ncs \
        --model-store $export_path \
        --models qa_server.mar \
        --ts-config config/config_template.properties
    sleep 1.5

    curl --fail -X POST "localhost:18081/models?url=$model_name.mar&$others"

    curl localhost:18080/ping
    if [ $? -eq 0 ]; then
        echo 'Finish packing the model!'
    else
        echo 'Fail'
    fi
 
}

main

