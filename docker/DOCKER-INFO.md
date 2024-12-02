## Pull from DockrHub:
```shell
docker pull shimonmal/memories_forgotten_concepts
```
## Optional - Build:
```shell
cd $HOME
git clone https://github.com/matanr/Memories_of_Forgotten_Concepts
cd Memories_of_Forgotten_Concepts
docker build -t memories_forgotten_concepts -f docker/Dockerfile .
```
## Optional - push to dockerhub:
(replace ```shimonmal``` with your dockerhub login name)
```shell
docker tag memories_forgotten_concepts shimonmal/memories_forgotten_concepts
docker push shimonmal/memories_forgotten_concepts
```

# Run locally:
```shell
docker run -it --gpus all -v $(pwd):/workspace memories_forgotten_concepts
```
and Run the appropriate command inside the docker container (see [running instructions](../README.md#running)).
