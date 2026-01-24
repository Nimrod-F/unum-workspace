gcloud functions deploy wordcount-unum-map-0 \
--runtime python313 \
--timeout 540s \
--trigger-topic wordcount-unum-map-0 \
--entry-point lambda_handler \
--env-vars-file env.yaml 