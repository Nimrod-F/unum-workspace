gcloud functions deploy iot-pipeline-aggregator \
--runtime python313 \
--trigger-topic iot-pipeline-aggregator \
--entry-point lambda_handler \
--env-vars-file env.yaml 