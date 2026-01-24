gcloud functions deploy text-processing-publish \
--runtime python313 \
--trigger-topic text-processing-publish \
--entry-point lambda_handler \
--env-vars-file env.yaml 