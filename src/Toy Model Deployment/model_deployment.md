# Deployment Instructions for Scikit Learn Example

1. clearml-serving create --name "serving example" --> Capture Output
2. clearml-serving --id 4bcb4d0d19c747f09409893246aafb5c model add --engine sklearn --endpoint "test_model_sklearn" --preprocess "examples/sklearn/preprocess.py" --name "train sklearn model" --project "serving examples"
3. cd C:\Users\natte\OneDrive\Studium\Master\2.Semester\MlOps\clearml-serving
4. 
5. docker run -v ~/clearml.conf:/root/clearml.conf -p 8080:8080 -e CLEARML_SERVING_TASK_ID=4bcb4d0d19c747f09409893246aafb5c -e CLEARML_SERVING_POLL_FREQ=5 clearml-serving-inference:latest