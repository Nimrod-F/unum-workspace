gcloud functions deploy text-processing-find-url \
--runtime python313 \
--trigger-topic text-processing-find-url \
--entry-point lambda_handler \
--env-vars-file env.yaml 