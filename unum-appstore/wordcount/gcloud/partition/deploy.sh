gcloud functions deploy wordcount-partition \
--runtime python313 \
--timeout 540s \
--trigger-topic wordcount-partition \
--entry-point lambda_handler \
--env-vars-file env.yaml 