gcloud functions deploy wordcount-reducer \
--runtime python313 \
--timeout 540s \
--trigger-topic wordcount-reducer \
--entry-point lambda_handler \
--env-vars-file env.yaml 