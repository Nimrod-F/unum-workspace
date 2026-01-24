gcloud functions deploy text-processing-shorten-url \
--runtime python313 \
--trigger-topic text-processing-shorten-url \
--entry-point lambda_handler \
--env-vars-file env.yaml 